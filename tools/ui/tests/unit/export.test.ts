import { describe, it, expect } from 'vitest';
import { exportConversationAsMarkdown, exportConversationAsHtml } from '$lib/utils/export';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';

function makeConv(): DatabaseConversation {
	return { id: 'c1', name: 'Test Chat', lastModified: Date.now(), currNode: 'm2' };
}

function makeMessages(): DatabaseMessage[] {
	return [
		{ id: 'm1', convId: 'c1', type: 'text', timestamp: 1000, role: 'user', content: 'Hello world', parent: null, children: ['m2'] },
		{ id: 'm2', convId: 'c1', type: 'text', timestamp: 2000, role: 'assistant', content: 'Hi there!', parent: 'm1', children: [] }
	];
}

describe('exportConversationAsMarkdown', () => {
	it('includes conversation name as heading', () => {
		const md = exportConversationAsMarkdown(makeConv(), makeMessages());
		expect(md).toContain('# Test Chat');
	});

	it('includes user and assistant messages', () => {
		const md = exportConversationAsMarkdown(makeConv(), makeMessages());
		expect(md).toContain('Hello world');
		expect(md).toContain('Hi there!');
	});

	it('skips root messages', () => {
		const msgs = [
			{ id: 'r1', convId: 'c1', type: 'root', timestamp: 0, role: 'system', content: '', parent: null, children: ['m1'] },
			...makeMessages()
		];
		const md = exportConversationAsMarkdown(makeConv(), msgs);
		expect(md).not.toContain('### System');
	});

	it('includes reasoning content in details block', () => {
		const msgs = [
			{ id: 'm1', convId: 'c1', type: 'text', timestamp: 1000, role: 'user', content: 'Q', parent: null, children: ['m2'] },
			{ id: 'm2', convId: 'c1', type: 'text', timestamp: 2000, role: 'assistant', content: 'Answer', reasoningContent: 'Let me think...', parent: 'm1', children: [] }
		];
		const md = exportConversationAsMarkdown(makeConv(), msgs);
		expect(md).toContain('Let me think...');
		expect(md).toContain('<details>');
	});
});

describe('exportConversationAsHtml', () => {
	it('returns valid HTML document', () => {
		const html = exportConversationAsHtml(makeConv(), makeMessages());
		expect(html).toContain('<!DOCTYPE html>');
		expect(html).toContain('</html>');
		expect(html).toContain('Hello world');
	});

	it('escapes HTML characters', () => {
		const conv = makeConv();
		conv.name = '<script>alert("xss")</script>';
		const msgs = [
			{ id: 'm1', convId: 'c1', type: 'text', timestamp: 1000, role: 'user', content: '<b>bold</b>', parent: null, children: ['m2'] },
			{ id: 'm2', convId: 'c1', type: 'text', timestamp: 2000, role: 'assistant', content: 'safe', parent: 'm1', children: [] }
		];
		const html = exportConversationAsHtml(conv, msgs);
		expect(html).not.toContain('<script>');
		expect(html).toContain('&lt;script&gt;');
		expect(html).not.toContain('<b>bold</b>');
		expect(html).toContain('&lt;b&gt;bold&lt;/b&gt;');
	});
});
