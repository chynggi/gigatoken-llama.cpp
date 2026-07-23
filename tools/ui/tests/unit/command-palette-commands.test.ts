import { describe, it, expect } from 'vitest';
import {
	buildCommandPaletteItems,
	filterCommandPaletteItems,
	groupCommandPaletteItems
} from '$lib/utils/command-palette-commands';

const routes = {
	start: '#/',
	newChat: '?new_chat=true#/',
	settings: '#/settings',
	settingsPacks: '#/settings/packs',
	skills: '#/skills',
	presets: '#/presets',
	searchProviders: '#/search-providers',
	chatHref: (id: string) => `#/chat/${id}`
};

describe('buildCommandPaletteItems', () => {
	it('includes non-archived conversations and navigation', () => {
		const items = buildCommandPaletteItems({
			conversations: [
				{ id: '1', name: 'Open me' },
				{ id: '2', name: 'Hidden', archived: true }
			],
			skillsEnabled: false,
			skills: [],
			presetsEnabled: false,
			presets: [],
			webSearchEnabled: false,
			routes
		});

		const labels = items.map((i) => i.label);
		expect(labels).toContain('Open me');
		expect(labels).not.toContain('Hidden');
		expect(labels).toContain('Settings');
		expect(labels).toContain('New conversation');
		expect(items.find((i) => i.id === 'conv-1')?.action).toEqual({
			type: 'goto',
			href: '#/chat/1'
		});
	});

	it('adds skill compose actions when skills enabled', () => {
		const items = buildCommandPaletteItems({
			conversations: [],
			skillsEnabled: true,
			skills: [{ id: 's1', name: 'summarize', description: 'Sum it' }],
			presetsEnabled: false,
			presets: [],
			webSearchEnabled: false,
			routes
		});

		const skill = items.find((i) => i.id === 'skill-s1');
		expect(skill?.action).toEqual({
			type: 'compose',
			text: '/summarize ',
			href: '#/'
		});
		expect(items.some((i) => i.id === 'manage-skills')).toBe(true);
	});

	it('adds preset apply and web search manage when enabled', () => {
		const items = buildCommandPaletteItems({
			conversations: [],
			skillsEnabled: false,
			skills: [],
			presetsEnabled: true,
			presets: [{ id: 'p1', name: 'Coding' }],
			webSearchEnabled: true,
			routes
		});

		expect(items.find((i) => i.id === 'preset-p1')?.action).toEqual({
			type: 'apply-preset',
			presetId: 'p1'
		});
		expect(items.some((i) => i.id === 'manage-search-providers')).toBe(true);
	});
});

describe('filterCommandPaletteItems / groupCommandPaletteItems', () => {
	const items = buildCommandPaletteItems({
		conversations: [{ id: '1', name: 'Alpha chat' }],
		skillsEnabled: true,
		skills: [{ id: 's', name: 'translate', description: 'lang tools' }],
		presetsEnabled: false,
		presets: [],
		webSearchEnabled: false,
		routes
	});

	it('filters by label and description', () => {
		expect(filterCommandPaletteItems(items, 'alpha').map((i) => i.id)).toContain('conv-1');
		expect(filterCommandPaletteItems(items, 'lang').map((i) => i.id)).toContain('skill-s');
		expect(filterCommandPaletteItems(items, 'zzzz').length).toBe(0);
	});

	it('groups by category', () => {
		const groups = groupCommandPaletteItems(items);
		expect(groups['Conversations']?.length).toBe(1);
		expect(groups['Skills']?.length).toBe(1);
		expect(groups['Navigation']?.length).toBeGreaterThan(0);
	});
});
