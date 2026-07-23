import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
	mapDuckDuckGoInstantAnswer,
	buildDuckDuckGoInstantAnswerUrl
} from '$lib/utils/ddgs';

describe('buildDuckDuckGoInstantAnswerUrl', () => {
	it('points at Instant Answer JSON API with format=json', () => {
		const url = buildDuckDuckGoInstantAnswerUrl('svelte 5 runes');
		expect(url.startsWith('https://api.duckduckgo.com/?')).toBe(true);
		expect(url).toContain('format=json');
		expect(url).toContain('no_html=1');
		expect(url).toMatch(/q=svelte[+%20]5[+%20]runes/);
		// No Lite HTML scrape endpoint
		expect(url).not.toContain('lite.duckduckgo.com');
	});
});

describe('mapDuckDuckGoInstantAnswer', () => {
	it('maps Abstract + RelatedTopics into title/url/snippet', () => {
		const fixture = {
			Heading: 'Svelte',
			AbstractText: 'A UI framework',
			AbstractURL: 'https://svelte.dev/',
			RelatedTopics: [
				{
					FirstURL: 'https://example.com/a',
					Text: 'Topic A - Details about A'
				},
				{
					Name: 'Group',
					Topics: [
						{
							FirstURL: 'https://example.com/b',
							Text: 'Topic B - Nested hit'
						}
					]
				}
			]
		};

		const results = mapDuckDuckGoInstantAnswer(fixture, 5);
		expect(results.length).toBeGreaterThanOrEqual(2);
		expect(results[0]).toMatchObject({
			title: 'Svelte',
			url: 'https://svelte.dev/',
			snippet: 'A UI framework'
		});
		expect(results.some((r) => r.url === 'https://example.com/a')).toBe(true);
		expect(results.some((r) => r.url === 'https://example.com/b')).toBe(true);
		for (const r of results) {
			expect(r).toHaveProperty('title');
			expect(r).toHaveProperty('url');
			expect(r).toHaveProperty('snippet');
		}
	});

	it('soft-fails to empty array on bad payload', () => {
		expect(mapDuckDuckGoInstantAnswer(null)).toEqual([]);
		expect(mapDuckDuckGoInstantAnswer({})).toEqual([]);
		expect(mapDuckDuckGoInstantAnswer('nope')).toEqual([]);
	});

	it('respects limit and dedupes urls', () => {
		const fixture = {
			RelatedTopics: [
				{ FirstURL: 'https://example.com/x', Text: 'One - a' },
				{ FirstURL: 'https://example.com/x', Text: 'Dup - b' },
				{ FirstURL: 'https://example.com/y', Text: 'Two - c' },
				{ FirstURL: 'https://example.com/z', Text: 'Three - d' }
			]
		};
		const results = mapDuckDuckGoInstantAnswer(fixture, 2);
		expect(results.length).toBe(2);
		expect(new Set(results.map((r) => r.url)).size).toBe(2);
	});
});

describe('SEARCH_PROVIDERS.ddgs shipped path', () => {
	it('wires Instant Answer mapper and has no Lite HTML scrape', () => {
		// Drive the shipped search.ts source (not a reimplementation)
		const searchPath = join(process.cwd(), 'src/lib/utils/search.ts');
		const source = readFileSync(searchPath, 'utf8');
		expect(source).toContain("id: 'ddgs'");
		expect(source).toContain('mapDuckDuckGoInstantAnswer');
		expect(source).toContain('buildDuckDuckGoInstantAnswerUrl');
		expect(source).not.toContain('lite.duckduckgo.com');
		// Old scrape regex must be gone
		expect(source).not.toMatch(/linkRe\s*=\s*\//);
	});
});
