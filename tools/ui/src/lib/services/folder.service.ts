import { IDXDB_TABLES } from '$lib/constants';
import { uuid } from '$lib/utils';
import { db } from './database.service';

export class FolderService {
	static async getAll(): Promise<DatabaseFolder[]> {
		return await db[IDXDB_TABLES.folders].orderBy('order').toArray();
	}

	static async get(id: string): Promise<DatabaseFolder | undefined> {
		return await db[IDXDB_TABLES.folders].get(id);
	}

	static async create(name: string, color?: string): Promise<DatabaseFolder> {
		const all = await db[IDXDB_TABLES.folders].orderBy('order').toArray();
		const folder: DatabaseFolder = {
			id: uuid(),
			name,
			color,
			order: all.length > 0 ? Math.max(...all.map((f) => f.order)) + 1 : 0,
			createdAt: Date.now()
		};
		await db[IDXDB_TABLES.folders].add(folder);
		return folder;
	}

	static async update(id: string, updates: Partial<Omit<DatabaseFolder, 'id'>>): Promise<void> {
		await db[IDXDB_TABLES.folders].update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		await db.transaction(
			'rw',
			[db[IDXDB_TABLES.folders], db[IDXDB_TABLES.conversations]],
			async () => {
				await db[IDXDB_TABLES.folders].delete(id);
				// Dexie ignores undefined in update(); clear membership via modify
				await db[IDXDB_TABLES.conversations]
					.filter((c) => c.folderId === id)
					.modify((c) => {
						delete c.folderId;
						c.lastModified = Date.now();
					});
			}
		);
	}
}
