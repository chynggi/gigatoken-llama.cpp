import { presetsTable } from '$lib/fork/db/tables';
import type { DatabasePreset } from '$lib/fork/db/types';
import { uuid } from '$lib/utils';

export class PresetService {
	static async getAll(): Promise<DatabasePreset[]> {
		return await presetsTable.orderBy('createdAt').toArray();
	}

	static async get(id: string): Promise<DatabasePreset | undefined> {
		return await presetsTable.get(id);
	}

	static async create(preset: Omit<DatabasePreset, 'id' | 'createdAt'>): Promise<DatabasePreset> {
		const newPreset: DatabasePreset = {
			...preset,
			id: uuid(),
			createdAt: Date.now()
		};
		await presetsTable.add(newPreset);
		return newPreset;
	}

	static async update(id: string, updates: Partial<Omit<DatabasePreset, 'id'>>): Promise<void> {
		await presetsTable.update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		await presetsTable.delete(id);
	}
}
