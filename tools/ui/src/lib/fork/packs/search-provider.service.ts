import { searchProvidersTable } from '$lib/fork/db/tables';
import type { DatabaseSearchProvider } from '$lib/fork/db/types';
import { uuid } from '$lib/utils';

export class SearchProviderService {
	static async getAll(): Promise<DatabaseSearchProvider[]> {
		return await searchProvidersTable.orderBy('priority').toArray();
	}

	static async get(id: string): Promise<DatabaseSearchProvider | undefined> {
		return await searchProvidersTable.get(id);
	}

	static async getEnabled(): Promise<DatabaseSearchProvider[]> {
		return await searchProvidersTable.filter((p) => p.enabled).sortBy('priority');
	}

	static async create(
		provider: Omit<DatabaseSearchProvider, 'id' | 'createdAt'>
	): Promise<DatabaseSearchProvider> {
		const newProvider: DatabaseSearchProvider = {
			...provider,
			id: uuid(),
			createdAt: Date.now()
		};
		await searchProvidersTable.add(newProvider);
		return newProvider;
	}

	static async update(
		id: string,
		updates: Partial<Omit<DatabaseSearchProvider, 'id'>>
	): Promise<void> {
		await searchProvidersTable.update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		await searchProvidersTable.delete(id);
	}

	static async toggleEnabled(id: string): Promise<boolean> {
		const provider = await searchProvidersTable.get(id);
		if (!provider) throw new Error(`Search provider ${id} not found`);
		const newState = !provider.enabled;
		await searchProvidersTable.update(id, { enabled: newState });
		return newState;
	}
}
