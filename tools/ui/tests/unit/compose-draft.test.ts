import { describe, it, expect, beforeEach } from 'vitest';
import {
	setPendingComposeText,
	consumePendingComposeText,
	peekPendingComposeText,
	isNewChatComposeTarget
} from '$lib/utils/compose-draft';
import { draftMessagesStore } from '$lib/stores/draft-messages.svelte';

describe('setPendingComposeText', () => {
	beforeEach(() => {
		consumePendingComposeText();
		draftMessagesStore.clearDraftMessage(undefined);
		draftMessagesStore.clearDraftMessage('chat-1');
	});

	it('stores pending text and NEW_CHAT draft so afterNavigate restore can see skill fill', () => {
		setPendingComposeText('/summarize ');

		expect(peekPendingComposeText()).toBe('/summarize ');

		const draft = draftMessagesStore.getDraftMessage(undefined);
		expect(draft.message).toBe('/summarize ');
		expect(draft.files).toEqual([]);

		const consumed = consumePendingComposeText();
		expect(consumed).toBe('/summarize ');
		expect(peekPendingComposeText()).toBeNull();

		// Draft remains for afterNavigate even after pending is consumed
		expect(draftMessagesStore.getDraftMessage(undefined).message).toBe('/summarize ');
	});

	it('overwrites previous new-chat draft with latest compose handoff', () => {
		draftMessagesStore.saveDraftMessage(undefined, 'old draft', []);
		setPendingComposeText('/translate lang=en ');
		expect(draftMessagesStore.getDraftMessage(undefined).message).toBe('/translate lang=en ');
		// Isolate: do not touch other chat drafts
		draftMessagesStore.saveDraftMessage('chat-1', 'open chat text', []);
		setPendingComposeText('/email ');
		expect(draftMessagesStore.getDraftMessage('chat-1').message).toBe('open chat text');
		expect(draftMessagesStore.getDraftMessage(undefined).message).toBe('/email ');
	});
});

describe('isNewChatComposeTarget', () => {
	it('is true only for new-chat (undefined/empty id), false for /chat/[id]', () => {
		expect(isNewChatComposeTarget(undefined)).toBe(true);
		expect(isNewChatComposeTarget('')).toBe(true);
		expect(isNewChatComposeTarget('abc-123')).toBe(false);
		expect(isNewChatComposeTarget('chat-1')).toBe(false);
	});
});

describe('compose handoff scoping (shipped applyComposeHandoff rules)', () => {
	/**
	 * Mirrors use-draft-messages applyComposeHandoff: only consume/apply when
	 * isNewChatComposeTarget(chatId). Drives real compose-draft + draftMessagesStore.
	 */
	function tryApplyHandoff(chatId: string | undefined): {
		applied: boolean;
		message: string;
		draftForChat: string;
		draftForNew: string;
		pendingLeft: string | null;
	} {
		let message = '';
		if (!isNewChatComposeTarget(chatId)) {
			return {
				applied: false,
				message,
				draftForChat: draftMessagesStore.getDraftMessage(chatId).message,
				draftForNew: draftMessagesStore.getDraftMessage(undefined).message,
				pendingLeft: peekPendingComposeText()
			};
		}
		const pending = consumePendingComposeText();
		if (!pending) {
			return {
				applied: false,
				message,
				draftForChat: draftMessagesStore.getDraftMessage(chatId).message,
				draftForNew: draftMessagesStore.getDraftMessage(undefined).message,
				pendingLeft: peekPendingComposeText()
			};
		}
		message = pending;
		draftMessagesStore.saveDraftMessage(undefined, pending, []);
		return {
			applied: true,
			message,
			draftForChat: draftMessagesStore.getDraftMessage(chatId).message,
			draftForNew: draftMessagesStore.getDraftMessage(undefined).message,
			pendingLeft: peekPendingComposeText()
		};
	}

	beforeEach(() => {
		consumePendingComposeText();
		draftMessagesStore.clearDraftMessage(undefined);
		draftMessagesStore.clearDraftMessage('chat-1');
		draftMessagesStore.saveDraftMessage('chat-1', 'existing chat draft', []);
	});

	it('does not inject skill text into /chat/[id] when pending is set (START noop then open chat)', () => {
		// User on START picks skill; goto(#/) is noop so pending stays
		setPendingComposeText('/summarize ');
		expect(peekPendingComposeText()).toBe('/summarize ');

		// Navigate to existing conversation - must not apply handoff
		const result = tryApplyHandoff('chat-1');
		expect(result.applied).toBe(false);
		expect(result.message).toBe('');
		expect(result.draftForChat).toBe('existing chat draft');
		// Pending remains for a later new-chat visit; new-chat draft still has skill
		expect(result.pendingLeft).toBe('/summarize ');
		expect(result.draftForNew).toBe('/summarize ');
	});

	it('applies skill text only when chatId is new-chat', () => {
		setPendingComposeText('/translate ');
		const result = tryApplyHandoff(undefined);
		expect(result.applied).toBe(true);
		expect(result.message).toBe('/translate ');
		expect(result.draftForNew).toBe('/translate ');
		expect(result.pendingLeft).toBeNull();
		// Existing chat draft untouched
		expect(draftMessagesStore.getDraftMessage('chat-1').message).toBe('existing chat draft');
	});
});

describe('use-draft-messages handoff source', () => {
	it('guards applyComposeHandoff with isNewChatComposeTarget in shipped hook', async () => {
		const { readFileSync } = await import('node:fs');
		const { join } = await import('node:path');
		const src = readFileSync(
			join(process.cwd(), 'src/lib/hooks/use-draft-messages.svelte.ts'),
			'utf8'
		);
		expect(src).toContain('isNewChatComposeTarget');
		expect(src).toContain('applyComposeHandoff');
		// Must refuse handoff when a chat id is present
		expect(src).toMatch(/if\s*\(\s*!isNewChatComposeTarget\(chatId\)\s*\)/);
	});
});
