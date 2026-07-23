/**
 * searchProvidersStore - Reactive store for web search providers.
 *
 * Manages search provider configs in IndexedDB, provider selection,
 * and search execution. Integrates with SearchProviderService and
 * the search provider implementations.
 */

import { browser } from '$app/environment';
import { SearchProviderService } from '$lib/services/search-provider.service';
import { formatSearchResultsForContext, getSearchProvider } from '$lib/utils/search';
import { config } from '$lib/stores/settings.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import type { SearchResult } from '$lib/utils/search';
import type { SearchProviderConfig } from '$lib/utils/search';

class SearchProvidersStore {
	providers = $state<DatabaseSearchProvider[]>([]);
	isInitialized = $state(false);

	/** Current search results for preview */
	searchResults = $state<SearchResult[]>([]);
	isSearching = $state(false);
	lastQuery = $state('');

	get enabled(): boolean {
		return Boolean(config()[SETTINGS_KEYS.WEB_SEARCH_ENABLED]);
	}

	get activeProviderId(): string {
		return String(config()[SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER] || '');
	}

	get autoDetect(): boolean {
		return Boolean(config()[SETTINGS_KEYS.WEB_SEARCH_AUTO_DETECT]);
	}

	async init(): Promise<void> {
		if (!browser) return;
		if (this.isInitialized) return;
		try {
			await this.loadProviders();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize search providers:', error);
		}
	}

	async loadProviders(): Promise<void> {
		this.providers = await SearchProviderService.getAll();
	}

	async addProvider(
		data: Omit<DatabaseSearchProvider, 'id' | 'createdAt'>
	): Promise<DatabaseSearchProvider> {
		const provider = await SearchProviderService.create(data);
		this.providers = [...this.providers, provider];
		return provider;
	}

	async updateProvider(id: string, updates: Partial<DatabaseSearchProvider>): Promise<void> {
		await SearchProviderService.update(id, updates);
		const idx = this.providers.findIndex((p) => p.id === id);
		if (idx !== -1) {
			this.providers[idx] = { ...this.providers[idx], ...updates };
			this.providers = [...this.providers];
		}
	}

	async toggleProvider(id: string): Promise<void> {
		const enabled = await SearchProviderService.toggleEnabled(id);
		const idx = this.providers.findIndex((p) => p.id === id);
		if (idx !== -1) {
			this.providers[idx] = { ...this.providers[idx], enabled };
			this.providers = [...this.providers];
		}
	}

	async deleteProvider(id: string): Promise<void> {
		await SearchProviderService.delete(id);
		this.providers = this.providers.filter((p) => p.id !== id);
	}

	/**
	 * Whether the given user message should trigger an automatic web search.
	 * When auto-detect is on, only messages ending with `?` search.
	 * When auto-detect is off, every send searches while web search is enabled.
	 */
	shouldSearch(message: string): boolean {
		if (!this.enabled || !message.trim()) return false;
		if (this.autoDetect) return message.trim().endsWith('?');
		return true;
	}

	async search(query: string): Promise<SearchResult[]> {
		if (!query.trim()) return [];

		// Resolve provider for this request only - do not auto-write settings
		let providerId = this.activeProviderId;
		if (!providerId) {
			const enabled = await SearchProviderService.getEnabled();
			if (enabled.length === 0) throw new Error('No search provider configured');
			providerId = enabled[0].id;
		}

		const dbProvider = await SearchProviderService.get(providerId);
		if (!dbProvider || !dbProvider.enabled) throw new Error('Search provider not available');

		const impl = getSearchProvider(dbProvider.type);
		if (!impl) throw new Error(`Unknown search provider: ${dbProvider.type}`);

		this.isSearching = true;
		this.lastQuery = query;

		try {
			const settings = config();
			const providerConfig: SearchProviderConfig = {
				apiKey: dbProvider.apiKey,
				baseUrl: dbProvider.baseUrl,
				resultsCount: Number(settings[SETTINGS_KEYS.WEB_SEARCH_RESULTS_COUNT]) || 5
			};
			const results = await impl.search(query, providerConfig);
			this.searchResults = results;
			return results;
		} finally {
			this.isSearching = false;
		}
	}

	/** Search and return a context block ready to append to the user message. */
	async searchForChatContext(query: string): Promise<string> {
		const results = await this.search(query);
		return formatSearchResultsForContext(query, results);
	}

	/** Clear the current search results */
	clearResults(): void {
		this.searchResults = [];
		this.lastQuery = '';
	}
}

export const searchProvidersStore = new SearchProvidersStore();

if (browser) {
	searchProvidersStore.init();
}
