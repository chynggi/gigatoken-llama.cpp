/**
 * DuckDuckGo Instant Answer JSON mapper.
 *
 * Uses the public Instant Answer API (format=json) instead of scraping Lite HTML.
 * RelatedTopics may nest Topics; Abstract / Results provide top-level hits.
 * Network fetch stays in search.ts; this module only maps structured payloads.
 */

import type { SearchResult } from './search';

export interface DuckDuckGoInstantAnswer {
	AbstractText?: string;
	AbstractURL?: string;
	AbstractSource?: string;
	Heading?: string;
	Results?: DuckDuckGoResultRow[];
	RelatedTopics?: DuckDuckGoRelatedTopic[];
}

export interface DuckDuckGoResultRow {
	FirstURL?: string;
	Text?: string;
	Result?: string;
}

export type DuckDuckGoRelatedTopic =
	| DuckDuckGoResultRow
	| { Name?: string; Topics?: DuckDuckGoResultRow[] };

/**
 * Map Instant Answer JSON into SearchResult[].
 * Soft-fails to [] when the payload is empty or unrecognized.
 */
export function mapDuckDuckGoInstantAnswer(
	data: unknown,
	limit = 5
): SearchResult[] {
	if (!data || typeof data !== 'object') return [];

	const payload = data as DuckDuckGoInstantAnswer;
	const out: SearchResult[] = [];
	const seen = new Set<string>();

	const push = (title: string, url: string, snippet: string) => {
		if (!url.startsWith('http') || !title.trim()) return;
		if (seen.has(url)) return;
		seen.add(url);
		out.push({ title: title.trim(), url, snippet: snippet.trim() });
	};

	if (payload.AbstractURL && (payload.Heading || payload.AbstractText)) {
		push(
			payload.Heading || payload.AbstractSource || 'Result',
			payload.AbstractURL,
			payload.AbstractText || ''
		);
	}

	for (const row of payload.Results || []) {
		if (out.length >= limit) break;
		if (row.FirstURL && row.Text) {
			push(row.Text, row.FirstURL, row.Text);
		}
	}

	const walkRelated = (topics: DuckDuckGoRelatedTopic[] | undefined) => {
		if (!topics) return;
		for (const topic of topics) {
			if (out.length >= limit) return;
			if ('Topics' in topic && Array.isArray(topic.Topics)) {
				walkRelated(topic.Topics);
				continue;
			}
			const row = topic as DuckDuckGoResultRow;
			if (row.FirstURL && row.Text) {
				// Instant Answer Text is often "Title - snippet"
				const dash = row.Text.indexOf(' - ');
				const title = dash > 0 ? row.Text.slice(0, dash) : row.Text;
				const snippet = dash > 0 ? row.Text.slice(dash + 3) : row.Text;
				push(title, row.FirstURL, snippet);
			}
		}
	};

	walkRelated(payload.RelatedTopics);

	return out.slice(0, limit);
}

/** Build Instant Answer API URL for a query. */
export function buildDuckDuckGoInstantAnswerUrl(query: string): string {
	const params = new URLSearchParams({
		q: query,
		format: 'json',
		no_html: '1',
		skip_disambig: '1'
	});
	return `https://api.duckduckgo.com/?${params.toString()}`;
}
