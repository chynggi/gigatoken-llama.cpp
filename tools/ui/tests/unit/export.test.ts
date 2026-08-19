import { exportConversationAsHtml, exportConversationAsMarkdown } from '$lib/fork/export/export';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import { describe, expect, it } from 'vitest';

const EXPORTED_AT = new Date('2026-08-15T12:00:00Z');

function conv(overrides: Partial<DatabaseConversation> = {}): DatabaseConversation {
	return {
		currNode: '',
		id: 'c1',
		lastModified: 0,
		name: 'My chat',
		...overrides
	};
}

function msg(overrides: Partial<DatabaseMessage> = {}): DatabaseMessage {
	return {
		children: [],
		content: 'hello',
		convId: 'c1',
		id: 'm1',
		parent: null,
		role: 'user',
		timestamp: 0,
		type: 'message',
		...overrides
	} as DatabaseMessage;
}

describe('exportConversationAsMarkdown', () => {
	it('renders a header, role labels, and a separator per message', () => {
		const out = exportConversationAsMarkdown(
			conv(),
			[msg({ content: 'hi' }), msg({ id: 'm2', role: 'assistant', content: 'hello back' })],
			EXPORTED_AT
		);

		expect(out).toContain('# My chat');
		expect(out).toContain(`_Exported on ${EXPORTED_AT.toLocaleString()}_`);
		expect(out).toContain('### You');
		expect(out).toContain('hi');
		expect(out).toContain('### Assistant');
		expect(out).toContain('hello back');
		expect(out.match(/^---$/gm)).toHaveLength(2);
	});

	it('skips root messages', () => {
		const out = exportConversationAsMarkdown(
			conv(),
			[msg({ type: 'root', content: 'ROOT' }), msg({ id: 'm2', content: 'real' })],
			EXPORTED_AT
		);

		expect(out).not.toContain('ROOT');
		expect(out).toContain('real');
		expect(out.match(/^---$/gm)).toHaveLength(1);
	});

	it('wraps reasoning content in a details block', () => {
		const out = exportConversationAsMarkdown(
			conv(),
			[msg({ role: 'assistant', reasoningContent: 'because' })],
			EXPORTED_AT
		);

		expect(out).toContain('<details>');
		expect(out).toContain('<summary>Thinking</summary>');
		expect(out).toContain('because');
		expect(out).toContain('</details>');
	});

	it('lists named attachments and falls back to a default title', () => {
		const out = exportConversationAsMarkdown(
			conv({ name: '' }),
			[msg({ extra: [{ name: 'notes.txt' }] as DatabaseMessage['extra'] })],
			EXPORTED_AT
		);

		expect(out).toContain('# Conversation');
		expect(out).toContain('_Attachment: notes.txt_');
	});

	it('labels unknown roles verbatim', () => {
		const out = exportConversationAsMarkdown(
			conv(),
			[msg({ role: 'tool' as DatabaseMessage['role'] })],
			EXPORTED_AT
		);

		expect(out).toContain('### tool');
	});
});

describe('exportConversationAsHtml', () => {
	it('produces a self-contained document with per-role classes', () => {
		const out = exportConversationAsHtml(
			conv(),
			[msg({ content: 'hi' }), msg({ id: 'm2', role: 'assistant', content: 'yo' })],
			EXPORTED_AT
		);

		expect(out.startsWith('<!DOCTYPE html>')).toBe(true);
		expect(out).toContain('<title>My chat</title>');
		expect(out).toContain(`Exported on ${EXPORTED_AT.toLocaleString()}`);
		expect(out).toContain('<div class="message user">');
		expect(out).toContain('<div class="message assistant">');
		expect(out).toContain('<style>');
	});

	it('escapes markup in the title, role, content, and attachments', () => {
		const out = exportConversationAsHtml(
			conv({ name: '<script>alert(1)</script>' }),
			[
				msg({
					content: 'a & b < c > d "q" \'s\'',
					extra: [{ name: '<img>' }] as DatabaseMessage['extra']
				})
			],
			EXPORTED_AT
		);

		expect(out).not.toContain('<script>alert(1)</script>');
		expect(out).toContain('&lt;script&gt;alert(1)&lt;/script&gt;');
		expect(out).toContain('a &amp; b &lt; c &gt; d &quot;q&quot; &#039;s&#039;');
		expect(out).toContain('Attachment: &lt;img&gt;');
	});

	it('converts newlines in content to line breaks', () => {
		const out = exportConversationAsHtml(conv(), [msg({ content: 'one\ntwo' })], EXPORTED_AT);

		expect(out).toContain('<div class="content">one<br>two</div>');
	});

	it('skips root messages and renders reasoning in a details block', () => {
		const out = exportConversationAsHtml(
			conv(),
			[
				msg({ type: 'root', content: 'ROOT' }),
				msg({ id: 'm2', role: 'assistant', content: 'ok', reasoningContent: 'why' })
			],
			EXPORTED_AT
		);

		expect(out).not.toContain('ROOT');
		expect(out).toContain('<details class="reasoning">');
		expect(out).toContain('<pre>why</pre>');
	});

	it('marks unknown roles with the other class', () => {
		const out = exportConversationAsHtml(
			conv(),
			[msg({ role: 'tool' as DatabaseMessage['role'] })],
			EXPORTED_AT
		);

		expect(out).toContain('<div class="message other">');
	});
});
