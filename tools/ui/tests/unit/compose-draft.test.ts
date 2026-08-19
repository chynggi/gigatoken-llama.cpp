import { isNewChatComposeTarget, setPendingComposeText } from '$lib/fork/command-palette/compose-draft';
import { draftMessagesStore } from '$lib/stores/draft-messages.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

/**
 * The palette hand-off deliberately owns no state of its own: it writes the
 * new-chat draft slot and lets upstream's `useDraftMessages` restore it on
 * navigation. These tests pin that contract.
 */

beforeEach(() => {
	draftMessagesStore.clearDraftMessage(undefined);
	draftMessagesStore.clearDraftMessage('chat-1');
});

describe('setPendingComposeText', () => {
	it('writes the new-chat draft slot so afterNavigate restore fills the compose box', () => {
		setPendingComposeText('/summarize ');

		expect(draftMessagesStore.getDraftMessage(undefined).message).toBe('/summarize ');
	});

	it('leaves an open chat draft untouched', () => {
		draftMessagesStore.saveDraftMessage('chat-1', 'work in progress', []);

		setPendingComposeText('/summarize ');

		expect(draftMessagesStore.getDraftMessage('chat-1').message).toBe('work in progress');
		expect(draftMessagesStore.getDraftMessage(undefined).message).toBe('/summarize ');
	});

	it('overwrites a previous hand-off with the latest one', () => {
		setPendingComposeText('/first ');
		setPendingComposeText('/second ');

		expect(draftMessagesStore.getDraftMessage(undefined).message).toBe('/second ');
	});

	it('queues no attachments alongside the text', () => {
		setPendingComposeText('/summarize ');

		expect(draftMessagesStore.getDraftMessage(undefined).files).toEqual([]);
	});
});

describe('isNewChatComposeTarget', () => {
	it('is true only for the new-chat route', () => {
		expect(isNewChatComposeTarget(undefined)).toBe(true);
		expect(isNewChatComposeTarget('')).toBe(true);
		expect(isNewChatComposeTarget('chat-1')).toBe(false);
	});
});
