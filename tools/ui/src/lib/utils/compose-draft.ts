/**
 * Pending chat compose text for cross-route handoff (e.g. command palette skills).
 *
 * Writes both:
 * - in-memory pending text (for same-page ChatForm listeners)
 * - NEW_CHAT draft slot (so afterNavigate draft restore does not wipe skill text)
 */

import { COMPOSE_CHAT_INPUT_EVENT } from '$lib/constants';
import { draftMessagesStore } from '$lib/stores/draft-messages.svelte';

let pendingComposeText: string | null = null;

function isBrowser(): boolean {
	return typeof document !== 'undefined';
}

/**
 * Queue compose text for the new-chat form.
 * Always persists into the new-chat draft key so navigation restore sees it.
 */
export function setPendingComposeText(text: string): void {
	pendingComposeText = text;
	// Target is ROUTES.START (undefined chat id -> NEW_CHAT_DRAFT_KEY)
	draftMessagesStore.saveDraftMessage(undefined, text, []);

	if (!isBrowser()) return;
	document.dispatchEvent(
		new CustomEvent(COMPOSE_CHAT_INPUT_EVENT, {
			detail: { text },
			bubbles: true
		})
	);
}

export function consumePendingComposeText(): string | null {
	const text = pendingComposeText;
	pendingComposeText = null;
	return text;
}

export function peekPendingComposeText(): string | null {
	return pendingComposeText;
}

/**
 * Palette skill handoff always targets the new-chat compose slot.
 * Existing /chat/[id] routes must not consume or apply pending text.
 */
export function isNewChatComposeTarget(chatId: string | undefined): boolean {
	return chatId === undefined || chatId === '';
}
