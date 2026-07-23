import { IDXDB_TABLES } from '$lib/constants';
import { uuid } from '$lib/utils';
import { db } from './database.service';

export class PresetService {
	static async getAll(): Promise<DatabasePreset[]> {
		return await db[IDXDB_TABLES.presets].orderBy('createdAt').toArray();
	}

	static async get(id: string): Promise<DatabasePreset | undefined> {
		return await db[IDXDB_TABLES.presets].get(id);
	}

	static async create(preset: Omit<DatabasePreset, 'id' | 'createdAt'>): Promise<DatabasePreset> {
		const newPreset: DatabasePreset = {
			...preset,
			id: uuid(),
			createdAt: Date.now()
		};
		await db[IDXDB_TABLES.presets].add(newPreset);
		return newPreset;
	}

	static async update(id: string, updates: Partial<Omit<DatabasePreset, 'id'>>): Promise<void> {
		await db[IDXDB_TABLES.presets].update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		await db[IDXDB_TABLES.presets].delete(id);
	}
}
