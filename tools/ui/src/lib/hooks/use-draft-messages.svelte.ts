import { onMount } from 'svelte';
import { afterNavigate, beforeNavigate } from '$app/navigation';
import { draftMessagesStore } from '$lib/stores/draft-messages.svelte';
import {
	consumePendingComposeText,
	isNewChatComposeTarget,
	peekPendingComposeText
} from '$lib/utils/compose-draft';

interface UseDraftMessagesOptions {
	getChatId: () => string | undefined;
	getMessage: () => string;
	getFiles: () => ChatUploadedFile[];
	setMessage: (message: string) => void;
	setFiles: (files: ChatUploadedFile[]) => void;
	getInitialMessage: () => string;
}

/**
 * Apply palette skill handoff only on the new-chat form (no route id).
 * Returns true when pending was applied and draft was written.
 */
function applyComposeHandoff(options: UseDraftMessagesOptions): boolean {
	const chatId = options.getChatId();
	// Do not inject skill text into /chat/[id] compose or that chat's draft
	if (!isNewChatComposeTarget(chatId)) {
		return false;
	}

	const pending = consumePendingComposeText();
	if (!pending) return false;

	options.setMessage(pending);
	options.setFiles([]);
	// undefined chatId -> NEW_CHAT_DRAFT_KEY
	draftMessagesStore.saveDraftMessage(undefined, pending, []);
	return true;
}

export function useDraftMessages(options: UseDraftMessagesOptions) {
	onMount(() => {
		if (applyComposeHandoff(options)) return;

		const chatId = options.getChatId();
		const draft = draftMessagesStore.getDraftMessage(chatId);

		if ((draft.message || draft.files.length > 0) && !options.getInitialMessage()) {
			options.setMessage(draft.message);
			options.setFiles(draft.files);
		}
	});

	beforeNavigate(() => {
		const chatId = options.getChatId();
		const message = options.getMessage();
		const pending = peekPendingComposeText();

		// Skill handoff temporarily sets the open-chat form value; do not pollute that chat's draft
		if (pending && message === pending && !isNewChatComposeTarget(chatId)) {
			return;
		}

		draftMessagesStore.saveDraftMessage(chatId, message, options.getFiles());
	});

	afterNavigate((navigation) => {
		if (navigation?.from == null) return;

		// Pending compose only for new-chat; existing chats load their own draft
		if (applyComposeHandoff(options)) return;

		const chatId = options.getChatId();
		const draft = draftMessagesStore.getDraftMessage(chatId);
		options.setMessage(draft.message);
		options.setFiles(draft.files);
	});

	function clearDraft() {
		const chatId = options.getChatId();
		draftMessagesStore.clearDraftMessage(chatId);
	}

	return { clearDraft };
}
