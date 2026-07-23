/**
 * Web Search provider abstraction and Tier 1 implementations.
 *
 * Each provider implements the SearchProvider interface. The active provider
 * is selected via searchProvidersStore; results are injected as chat context.
 */

import { buildProxiedHeaders, buildProxiedUrl } from '$lib/utils/cors-proxy';

export interface SearchResult {
	title: string;
	url: string;
	snippet: string;
	favicon?: string;
	publishedDate?: string;
}

export interface SearchProviderConfig {
	apiKey?: string;
	baseUrl?: string;
	resultsCount?: number;
}

/**
 * Base interface for all search providers.
 */
export interface SearchProvider {
	id: string;
	type: string;
	name: string;
	requiresApiKey: boolean;
	requiresBaseUrl: boolean;
	search(query: string, config: SearchProviderConfig): Promise<SearchResult[]>;
}

function headersToRecord(headers?: HeadersInit): Record<string, string> {
	if (!headers) return {};
	if (headers instanceof Headers) {
		return Object.fromEntries(headers.entries());
	}
	if (Array.isArray(headers)) {
		return Object.fromEntries(headers);
	}
	return { ...headers };
}

/**
 * Fetch via llama-server CORS proxy so third-party search APIs work in-browser.
 * Preserves method, headers, and body.
 */
async function proxyFetch(url: string, options?: RequestInit): Promise<Response> {
	const proxiedUrl = buildProxiedUrl(url);
	const headers = buildProxiedHeaders(headersToRecord(options?.headers));

	return fetch(proxiedUrl, {
		method: options?.method || 'GET',
		headers,
		body: options?.body,
		signal: options?.signal
	});
}

/**
 * Format search results for injection into the user message sent to the model.
 */
export function formatSearchResultsForContext(query: string, results: SearchResult[]): string {
	if (results.length === 0) return '';

	const lines = results.map((r, i) => {
		const snippet = r.snippet ? `\n   ${r.snippet}` : '';
		return `${i + 1}. [${r.title}](${r.url})${snippet}`;
	});

	return [
		'',
		'---',
		`Web search results for "${query}":`,
		...lines,
		'---',
		''
	].join('\n');
}

// ---------------------------------------------------------------------------
// Tier 1: SearXNG (self-hosted, no API key required)
// ---------------------------------------------------------------------------

export const searxngProvider: SearchProvider = {
	id: 'searxng',
	type: 'searxng',
	name: 'SearXNG',
	requiresApiKey: false,
	requiresBaseUrl: true,

	async search(query: string, config: SearchProviderConfig): Promise<SearchResult[]> {
		const baseUrl = config.baseUrl?.replace(/\/$/, '') || 'http://localhost:8080';
		const url = `${baseUrl}/search?q=${encodeURIComponent(query)}&format=json&categories=general`;

		// Prefer proxy for non-localhost instances; direct fetch for local SearXNG.
		const isLocal =
			baseUrl.includes('localhost') || baseUrl.includes('127.0.0.1') || baseUrl.includes('[::1]');
		const resp = isLocal ? await fetch(url) : await proxyFetch(url);
		if (!resp.ok) throw new Error(`SearXNG returned ${resp.status}`);

		const data = await resp.json();
		const results = data.results || [];

		return results.slice(0, config.resultsCount || 5).map((r: Record<string, unknown>) => ({
			title: String(r.title || ''),
			url: String(r.url || ''),
			snippet: String(r.content || r.snippet || ''),
			favicon: r.favicon ? String(r.favicon) : undefined,
			publishedDate: r.publishedDate ? String(r.publishedDate) : undefined
		}));
	}
};

// ---------------------------------------------------------------------------
// Tier 1: Tavily (AI-optimized search, free tier 1000/month)
// ---------------------------------------------------------------------------

export const tavilyProvider: SearchProvider = {
	id: 'tavily',
	type: 'tavily',
	name: 'Tavily',
	requiresApiKey: true,
	requiresBaseUrl: false,

	async search(query: string, config: SearchProviderConfig): Promise<SearchResult[]> {
		if (!config.apiKey) throw new Error('Tavily API key required');

		const resp = await proxyFetch('https://api.tavily.com/search', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				api_key: config.apiKey,
				query,
				search_depth: 'basic',
				max_results: config.resultsCount || 5
			})
		});

		if (!resp.ok) throw new Error(`Tavily returned ${resp.status}`);
		const data = await resp.json();
		return (data.results || []).map((r: Record<string, unknown>) => ({
			title: String(r.title || ''),
			url: String(r.url || ''),
			snippet: String(r.content || '')
		}));
	}
};

// ---------------------------------------------------------------------------
// Tier 1: Brave Search (free tier 2000/month)
// ---------------------------------------------------------------------------

export const braveProvider: SearchProvider = {
	id: 'brave',
	type: 'brave',
	name: 'Brave Search',
	requiresApiKey: true,
	requiresBaseUrl: false,

	async search(query: string, config: SearchProviderConfig): Promise<SearchResult[]> {
		if (!config.apiKey) throw new Error('Brave API key required');

		const count = config.resultsCount || 5;
		const resp = await proxyFetch(
			`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${count}`,
			{
				headers: {
					Accept: 'application/json',
					'Accept-Encoding': 'gzip',
					'X-Subscription-Token': config.apiKey
				}
			}
		);

		if (!resp.ok) throw new Error(`Brave returned ${resp.status}`);
		const data = await resp.json();
		return (data.web?.results || []).map((r: Record<string, unknown>) => {
			const meta = r.meta_url as Record<string, string> | undefined;
			return {
				title: String(r.title || ''),
				url: String(r.url || ''),
				snippet: String(r.description || ''),
				favicon: meta?.favicon ? String(meta.favicon) : undefined,
				publishedDate: r.age ? String(r.age) : undefined
			};
		});
	}
};

// ---------------------------------------------------------------------------
// Tier 1: Serper (Google results, free tier 2500/month)
// ---------------------------------------------------------------------------

export const serperProvider: SearchProvider = {
	id: 'serper',
	type: 'serper',
	name: 'Serper',
	requiresApiKey: true,
	requiresBaseUrl: false,

	async search(query: string, config: SearchProviderConfig): Promise<SearchResult[]> {
		if (!config.apiKey) throw new Error('Serper API key required');

		const resp = await proxyFetch('https://google.serper.dev/search', {
			method: 'POST',
			headers: { 'X-API-KEY': config.apiKey, 'Content-Type': 'application/json' },
			body: JSON.stringify({ q: query, num: config.resultsCount || 5 })
		});

		if (!resp.ok) throw new Error(`Serper returned ${resp.status}`);
		const data = await resp.json();
		return (data.organic || []).map((r: Record<string, unknown>) => ({
			title: String(r.title || ''),
			url: String(r.link || ''),
			snippet: String(r.snippet || ''),
			publishedDate: r.date ? String(r.date) : undefined
		}));
	}
};

// ---------------------------------------------------------------------------
// Tier 1: DDGS via DuckDuckGo Instant Answer JSON (no HTML scrape)
// ---------------------------------------------------------------------------

export const ddgsProvider: SearchProvider = {
	id: 'ddgs',
	type: 'ddgs',
	name: 'DuckDuckGo',
	requiresApiKey: false,
	requiresBaseUrl: false,

	async search(query: string, config: SearchProviderConfig): Promise<SearchResult[]> {
		// Instant Answer API returns structured JSON (Abstract / RelatedTopics / Results).
		// Coverage is topic-oriented rather than full web SERP; empty results are a soft fail.
		const { buildDuckDuckGoInstantAnswerUrl, mapDuckDuckGoInstantAnswer } = await import('./ddgs');
		const url = buildDuckDuckGoInstantAnswerUrl(query);
		const resp = await proxyFetch(url, {
			headers: { Accept: 'application/json' }
		});

		if (!resp.ok) throw new Error(`DDGS returned ${resp.status}`);

		let data: unknown;
		try {
			data = await resp.json();
		} catch {
			return [];
		}

		return mapDuckDuckGoInstantAnswer(data, config.resultsCount || 5);
	}
};

// ---------------------------------------------------------------------------
// Provider registry
// ---------------------------------------------------------------------------

export const SEARCH_PROVIDERS: Record<SearchProviderType, SearchProvider> = {
	searxng: searxngProvider,
	tavily: tavilyProvider,
	brave: braveProvider,
	serper: serperProvider,
	ddgs: ddgsProvider
};

export function getSearchProvider(type: string): SearchProvider | undefined {
	return SEARCH_PROVIDERS[type as SearchProviderType];
}
