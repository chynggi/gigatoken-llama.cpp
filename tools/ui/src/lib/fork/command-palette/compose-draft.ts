/**
 * Chat compose hand-off for the command palette.
 *
 * Picking a skill in the palette queues its text for a *new* chat by writing
 * the new-chat draft slot. Upstream's `useDraftMessages` restores that slot in
 * `afterNavigate`, so the palette only has to save and navigate — no hook of
 * our own is needed inside the upstream draft machinery.
 *
 * Writing the new-chat slot (rather than the active chat's) is what keeps skill
 * text out of an open `/chat/[id]` compose box.
 */

import { draftMessagesStore } from '$lib/stores';

/**
 * Queue compose text for the new-chat form.
 *
 * @param text - Text to place in the next new chat's compose box
 */
export function setPendingComposeText(text: string): void {
	// `undefined` chat id targets the NEW_CHAT draft key.
	draftMessagesStore.saveDraftMessage(undefined, text, []);
}

/**
 * Whether a chat id refers to the new-chat route, which is the only target the
 * palette hand-off is allowed to fill.
 *
 * @param chatId - Route chat id, or `undefined` on the new-chat route
 */
export function isNewChatComposeTarget(chatId: string | undefined): boolean {
	return chatId === undefined || chatId === '';
}
