import { IDXDB_TABLES } from '$lib/constants';
import { uuid } from '$lib/utils';
import { db } from './database.service';

export class SearchProviderService {
	static async getAll(): Promise<DatabaseSearchProvider[]> {
		return await db[IDXDB_TABLES.searchProviders].orderBy('priority').toArray();
	}

	static async get(id: string): Promise<DatabaseSearchProvider | undefined> {
		return await db[IDXDB_TABLES.searchProviders].get(id);
	}

	static async getEnabled(): Promise<DatabaseSearchProvider[]> {
		return await db[IDXDB_TABLES.searchProviders].filter((p) => p.enabled).sortBy('priority');
	}

	static async create(
		provider: Omit<DatabaseSearchProvider, 'id' | 'createdAt'>
	): Promise<DatabaseSearchProvider> {
		const newProvider: DatabaseSearchProvider = {
			...provider,
			id: uuid(),
			createdAt: Date.now()
		};
		await db[IDXDB_TABLES.searchProviders].add(newProvider);
		return newProvider;
	}

	static async update(
		id: string,
		updates: Partial<Omit<DatabaseSearchProvider, 'id'>>
	): Promise<void> {
		await db[IDXDB_TABLES.searchProviders].update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		await db[IDXDB_TABLES.searchProviders].delete(id);
	}

	static async toggleEnabled(id: string): Promise<boolean> {
		const provider = await db[IDXDB_TABLES.searchProviders].get(id);
		if (!provider) throw new Error(`Search provider ${id} not found`);
		const newState = !provider.enabled;
		await db[IDXDB_TABLES.searchProviders].update(id, { enabled: newState });
		return newState;
	}
}
