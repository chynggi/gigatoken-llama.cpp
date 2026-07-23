import { describe, it, expect } from 'vitest';
import {
	parseSkillCommand,
	applySkillTemplate,
	missingPlaceholders,
	rankSkills
} from '$lib/utils/skill-engine';
import type { DatabaseSkill } from '$lib/types/database';

function makeSkill(overrides: Partial<DatabaseSkill> = {}): DatabaseSkill {
	return {
		id: 'test-1',
		name: 'test-skill',
		description: 'A test skill',
		content: 'Hello {{name}}',
		category: 'writing',
		placeholders: [{ name: 'name', description: 'Your name' }],
		isBuiltIn: false,
		createdAt: Date.now(),
		...overrides
	};
}

describe('parseSkillCommand', () => {
	it('returns null for non-slash input', () => {
		expect(parseSkillCommand('hello')).toBeNull();
		expect(parseSkillCommand('')).toBeNull();
	});

	it('returns null for just a slash', () => {
		expect(parseSkillCommand('/')).toBeNull();
	});

	it('parses skill name without arguments', () => {
		const result = parseSkillCommand('/summarize');
		expect(result).not.toBeNull();
		expect(result!.skillName).toBe('summarize');
		expect(result!.args).toEqual({});
	});

	it('parses skill name with multiple key=value arguments', () => {
		const result = parseSkillCommand('/translate lang=ko text=Hello world');
		expect(result).not.toBeNull();
		expect(result!.skillName).toBe('translate');
		expect(result!.args).toEqual({ lang: 'ko', text: 'Hello world' });
	});

	it('handles trailing text as "text" argument when no key=value', () => {
		const result = parseSkillCommand('/summarize this is some text');
		expect(result).not.toBeNull();
		expect(result!.skillName).toBe('summarize');
		expect(result!.args).toEqual({ text: 'this is some text' });
	});

	it('captures trailing text as part of single key=value (greedy)', () => {
		const result = parseSkillCommand('/rewrite style=formal Make this better');
		expect(result).not.toBeNull();
		expect(result!.skillName).toBe('rewrite');
		expect(result!.args).toEqual({ style: 'formal Make this better' });
	});

	it('mixes multiple key=value args with explicit keys', () => {
		const result = parseSkillCommand('/rewrite text=Make this better style=formal');
		expect(result).not.toBeNull();
		expect(result!.skillName).toBe('rewrite');
		expect(result!.args).toEqual({ text: 'Make this better', style: 'formal' });
	});

	it('handles empty argument value', () => {
		const result = parseSkillCommand('/email topic=');
		expect(result).not.toBeNull();
		expect(result!.args).toEqual({ topic: '' });
	});

	it('handles argument with spaces around equals', () => {
		const result = parseSkillCommand('/translate lang = ko');
		expect(result).not.toBeNull();
		expect(result!.skillName).toBe('translate');
		expect(result!.args).toEqual({ lang: 'ko' });
	});

	it('handles quoted argument values and strips quotes', () => {
		const result = parseSkillCommand('/email topic="Hello world" tone=formal');
		expect(result).not.toBeNull();
		expect(result!.args).toEqual({ topic: 'Hello world', tone: 'formal' });
	});
});

describe('applySkillTemplate', () => {
	it('substitutes placeholders with provided args', () => {
		const skill = makeSkill({
			content: 'Hello {{name}}, welcome to {{place}}',
			placeholders: [
				{ name: 'name', description: 'Name' },
				{ name: 'place', description: 'Place' }
			]
		});
		const result = applySkillTemplate(skill, { name: 'World', place: 'Korea' });
		expect(result).toBe('Hello World, welcome to Korea');
	});

	it('uses default values when arg is missing', () => {
		const skill = makeSkill({
			content: 'Translate to {{lang}}: {{text}}',
			placeholders: [
				{ name: 'lang', description: 'Language', defaultValue: 'en' },
				{ name: 'text', description: 'Text' }
			]
		});
		const result = applySkillTemplate(skill, { text: 'Hello' });
		expect(result).toBe('Translate to en: Hello');
	});

	it('removes unmatched placeholders', () => {
		const skill = makeSkill({
			content: 'Hello {{name}}, this is {{unknown}}',
			placeholders: [{ name: 'name', description: 'Name' }]
		});
		const result = applySkillTemplate(skill, { name: 'World' });
		expect(result).toBe('Hello World, this is ');
	});

	it('handles no placeholders defined', () => {
		const skill = makeSkill({
			content: 'Hello {{name}}',
			placeholders: undefined
		});
		const result = applySkillTemplate(skill, { name: 'World' });
		expect(result).toBe('Hello World');
	});

	it('handles empty args', () => {
		const skill = makeSkill({
			content: 'Hello {{name}}',
			placeholders: [{ name: 'name', description: 'Name', defaultValue: 'there' }]
		});
		const result = applySkillTemplate(skill, {});
		expect(result).toBe('Hello there');
	});
});

describe('missingPlaceholders', () => {
	it('returns placeholders without defaults when no args provided', () => {
		const skill = makeSkill({
			placeholders: [
				{ name: 'a', description: 'A' },
				{ name: 'b', description: 'B', defaultValue: 'default' }
			]
		});
		const missing = missingPlaceholders(skill, {});
		expect(missing).toHaveLength(1);
		expect(missing[0].name).toBe('a');
	});

	it('returns empty when all required placeholders have values', () => {
		const skill = makeSkill({
			placeholders: [
				{ name: 'a', description: 'A' },
				{ name: 'b', description: 'B' }
			]
		});
		const missing = missingPlaceholders(skill, { a: '1', b: '2' });
		expect(missing).toHaveLength(0);
	});

	it('returns empty for undefined placeholders', () => {
		const skill = makeSkill({ placeholders: undefined });
		expect(missingPlaceholders(skill, {})).toHaveLength(0);
	});
});

describe('rankSkills', () => {
	const skills: DatabaseSkill[] = [
		makeSkill({ id: '1', name: 'summarize', description: 'Summarize text', usageCount: 10 }),
		makeSkill({ id: '2', name: 'translate', description: 'Translate text', usageCount: 20 }),
		makeSkill({ id: '3', name: 'summary', description: 'Create summary', usageCount: 5 }),
		makeSkill({ id: '4', name: 'rewrite', description: 'Summarize and rewrite', usageCount: 0 })
	];

	it('returns exact match first', () => {
		const result = rankSkills(skills, 'summarize');
		expect(result[0].name).toBe('summarize');
	});

	it('returns prefix matches before contains matches', () => {
		const result = rankSkills(skills, 'sum');
		expect(result[0].name).toBe('summarize');
		expect(result.length).toBeGreaterThanOrEqual(2);
	});

	it('ranks by description match lower', () => {
		const result = rankSkills(skills, 'summar');
		const names = result.map((s) => s.name);
		expect(names).toContain('rewrite');
		expect(names.indexOf('summarize')).toBeLessThan(names.indexOf('rewrite'));
	});

	it('returns empty for no matches', () => {
		const result = rankSkills(skills, 'zzz');
		expect(result).toHaveLength(0);
	});

	it('sorts by usage count within same tier', () => {
		const result = rankSkills(skills, 's');
		const summarizeIdx = result.findIndex((s) => s.name === 'summarize');
		const summaryIdx = result.findIndex((s) => s.name === 'summary');
		expect(summarizeIdx).toBeLessThan(summaryIdx);
	});
});
