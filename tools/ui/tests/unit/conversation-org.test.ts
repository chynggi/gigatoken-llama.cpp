import { describe, it, expect } from 'vitest';
import {
	patchConversationOrgState,
	collectAllTags,
	nextTagFilter,
	conversationIdsInFolder,
	tagsAfterAdd,
	tagsAfterRemove
} from '$lib/utils/conversation-org';

function conv(
	partial: Partial<DatabaseConversation> & Pick<DatabaseConversation, 'id' | 'name'>
): DatabaseConversation {
	return { lastModified: 1, currNode: '', ...partial };
}

describe('patchConversationOrgState', () => {
	it('updates list entry and active conversation when ids match', () => {
		const list = [conv({ id: 'a', name: 'A' }), conv({ id: 'b', name: 'B', folderId: 'f1' })];
		const active = list[1];
		const { conversations, activeConversation } = patchConversationOrgState(
			list,
			active,
			'b',
			{ tags: ['work'], archived: true }
		);

		expect(conversations[1].tags).toEqual(['work']);
		expect(conversations[1].archived).toBe(true);
		expect(activeConversation?.tags).toEqual(['work']);
		expect(activeConversation?.archived).toBe(true);
		// original list not mutated in place for the patched entry identity
		expect(list[1].tags).toBeUndefined();
	});

	it('clears folderId when patch sets undefined', () => {
		const list = [conv({ id: 'a', name: 'A', folderId: 'f1' })];
		const { conversations, activeConversation } = patchConversationOrgState(
			list,
			list[0],
			'a',
			{ folderId: undefined }
		);
		expect(conversations[0].folderId).toBeUndefined();
		expect('folderId' in conversations[0]).toBe(false);
		expect(activeConversation && 'folderId' in activeConversation).toBe(false);
	});

	it('does not change active when ids differ', () => {
		const list = [conv({ id: 'a', name: 'A' }), conv({ id: 'b', name: 'B' })];
		const active = list[0];
		const { activeConversation } = patchConversationOrgState(list, active, 'b', {
			archived: true
		});
		expect(activeConversation?.id).toBe('a');
		expect(activeConversation?.archived).toBeUndefined();
	});
});

describe('collectAllTags / tag helpers', () => {
	it('collects unique sorted tags', () => {
		const list = [
			conv({ id: '1', name: 'x', tags: ['b', 'a'] }),
			conv({ id: '2', name: 'y', tags: ['a', 'c'] })
		];
		expect(collectAllTags(list)).toEqual(['a', 'b', 'c']);
	});

	it('toggles tag filter', () => {
		expect(nextTagFilter(null, 'work')).toBe('work');
		expect(nextTagFilter('work', 'work')).toBeNull();
		expect(nextTagFilter('work', 'home')).toBe('home');
	});

	it('tagsAfterAdd and tagsAfterRemove', () => {
		expect(tagsAfterAdd(['a'], 'a')).toBeNull();
		expect(tagsAfterAdd(['a'], 'b')).toEqual(['a', 'b']);
		expect(tagsAfterRemove(['a', 'b'], 'a')).toEqual(['b']);
	});

	it('conversationIdsInFolder', () => {
		const list = [
			conv({ id: '1', name: 'x', folderId: 'f' }),
			conv({ id: '2', name: 'y' }),
			conv({ id: '3', name: 'z', folderId: 'f' })
		];
		expect(conversationIdsInFolder(list, 'f')).toEqual(['1', '3']);
	});
});
