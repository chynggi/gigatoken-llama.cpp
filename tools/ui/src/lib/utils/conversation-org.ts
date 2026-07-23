/**
 * Pure helpers for conversation organization (folder / tag / archive).
 * Used by conversationsStore so org mutations stay testable outside the store class.
 */

export type ConversationOrgPatch = Partial<
	Pick<DatabaseConversation, 'folderId' | 'tags' | 'archived'>
>;

/**
 * Apply an org metadata patch to a conversation list entry and optional active chat.
 * Clears folderId when the patch sets it to undefined (Dexie-compatible in-memory shape).
 */
export function patchConversationOrgState(
	conversations: DatabaseConversation[],
	activeConversation: DatabaseConversation | null,
	convId: string,
	patch: ConversationOrgPatch
): {
	conversations: DatabaseConversation[];
	activeConversation: DatabaseConversation | null;
} {
	const nextList = conversations.map((c) => {
		if (c.id !== convId) return c;
		return applyOrgPatch(c, patch);
	});

	let nextActive = activeConversation;
	if (activeConversation?.id === convId) {
		nextActive = applyOrgPatch(activeConversation, patch);
	}

	return { conversations: nextList, activeConversation: nextActive };
}

function applyOrgPatch(
	conv: DatabaseConversation,
	patch: ConversationOrgPatch
): DatabaseConversation {
	const next: DatabaseConversation = { ...conv, ...patch };
	if ('folderId' in patch && patch.folderId === undefined) {
		delete next.folderId;
	}
	return next;
}

/** Unique sorted tags across all conversations. */
export function collectAllTags(conversations: DatabaseConversation[]): string[] {
	const tagSet = new Set<string>();
	for (const conv of conversations) {
		if (!conv.tags) continue;
		for (const t of conv.tags) tagSet.add(t);
	}
	return [...tagSet].sort();
}

/** Toggle tag filter: same tag clears; different tag selects. */
export function nextTagFilter(current: string | null, tag: string | null): string | null {
	if (tag === null) return null;
	return current === tag ? null : tag;
}

/** Conversation ids that belong to a folder (for delete-folder cleanup). */
export function conversationIdsInFolder(
	conversations: DatabaseConversation[],
	folderId: string
): string[] {
	return conversations.filter((c) => c.folderId === folderId).map((c) => c.id);
}

/** Build new tags array after add (no-op if duplicate). */
export function tagsAfterAdd(tags: string[] | undefined, tag: string): string[] | null {
	const current = tags ?? [];
	if (current.includes(tag)) return null;
	return [...current, tag];
}

/** Build new tags array after remove. */
export function tagsAfterRemove(tags: string[] | undefined, tag: string): string[] {
	return (tags ?? []).filter((t) => t !== tag);
}
