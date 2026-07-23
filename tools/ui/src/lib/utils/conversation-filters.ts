/**
 * Pure helpers for filtering conversations by org state and name search.
 */

export interface ConversationFilterOptions {
	searchQuery?: string;
	folderId?: string;
	tag?: string | null;
	showArchived?: boolean;
}

/**
 * Apply folder / tag / archive / name filters to a conversation list.
 * Archive filter is skipped when a non-empty search query is active so
 * search can find archived conversations by name.
 */
export function filterConversations(
	conversations: DatabaseConversation[],
	options: ConversationFilterOptions = {}
): DatabaseConversation[] {
	const { searchQuery = '', folderId, tag, showArchived = false } = options;
	const q = searchQuery.trim().toLowerCase();
	const isSearching = q.length > 0;

	return conversations.filter((conversation) => {
		if (!isSearching && !showArchived && conversation.archived) {
			return false;
		}

		if (folderId !== undefined && conversation.folderId !== folderId) {
			return false;
		}

		if (tag) {
			const tags = conversation.tags ?? [];
			if (!tags.includes(tag)) return false;
		}

		if (isSearching && !conversation.name.toLowerCase().includes(q)) {
			return false;
		}

		return true;
	});
}
