import { describe, it, expect } from 'vitest';
import { filterConversations } from '$lib/utils/conversation-filters';

function conv(
	partial: Partial<DatabaseConversation> & Pick<DatabaseConversation, 'id' | 'name'>
): DatabaseConversation {
	return {
		lastModified: Date.now(),
		currNode: '',
		...partial
	};
}

describe('filterConversations', () => {
	const items = [
		conv({ id: '1', name: 'Alpha chat', folderId: 'f1', tags: ['work'] }),
		conv({ id: '2', name: 'Beta notes', folderId: 'f2', archived: true }),
		conv({ id: '3', name: 'Gamma', tags: ['personal', 'work'] })
	];

	it('hides archived by default', () => {
		const result = filterConversations(items);
		expect(result.map((c) => c.id)).toEqual(['1', '3']);
	});

	it('shows archived when requested', () => {
		const result = filterConversations(items, { showArchived: true });
		expect(result.map((c) => c.id)).toEqual(['1', '2', '3']);
	});

	it('filters by folder', () => {
		const result = filterConversations(items, { folderId: 'f1' });
		expect(result.map((c) => c.id)).toEqual(['1']);
	});

	it('filters by tag', () => {
		const result = filterConversations(items, { tag: 'work' });
		expect(result.map((c) => c.id)).toEqual(['1', '3']);
	});

	it('filters by search query and includes archived', () => {
		const result = filterConversations(items, { searchQuery: 'beta' });
		expect(result.map((c) => c.id)).toEqual(['2']);
	});
});
