import {
	buildCommandPaletteItems,
	filterCommandPaletteItems,
	groupCommandPaletteItems
} from '$lib/fork/command-palette/command-palette-commands';
import { describe, expect, it } from 'vitest';

const routes = {
	chatHref: (id: string) => `#/chat/${id}`,
	newChat: '?new_chat=true#/',
	presets: '#/presets',
	searchProviders: '#/search-providers',
	skills: '#/skills',
	start: '#/'
};

describe('buildCommandPaletteItems', () => {
	it('includes non-archived conversations and navigation', () => {
		const items = buildCommandPaletteItems({
			conversations: [
				{ id: '1', name: 'Open me' },
				{ archived: true, id: '2', name: 'Hidden' }
			],
			presets: [],
			presetsEnabled: false,
			routes,
			skills: [],
			skillsEnabled: false,
			webSearchEnabled: false
		});
		const labels = items.map((i) => i.label);

		expect(labels).toContain('Open me');
		expect(labels).not.toContain('Hidden');
		expect(labels).toContain('Settings');
		expect(items.find((i) => i.id === 'settings-general')?.action).toEqual({
			type: 'open-settings'
		});
		expect(items.find((i) => i.id === 'settings-packs')?.action).toEqual({
			section: 'packs',
			type: 'open-settings'
		});
		expect(labels).toContain('New conversation');
		expect(items.find((i) => i.id === 'conv-1')?.action).toEqual({
			href: '#/chat/1',
			type: 'goto'
		});
	});

	it('adds skill compose actions when skills enabled', () => {
		const items = buildCommandPaletteItems({
			conversations: [],
			presets: [],
			presetsEnabled: false,
			routes,
			skills: [{ description: 'Sum it', id: 's1', name: 'summarize' }],
			skillsEnabled: true,
			webSearchEnabled: false
		});
		const skill = items.find((i) => i.id === 'skill-s1');

		expect(skill?.action).toEqual({
			href: '#/',
			text: '/summarize ',
			type: 'compose'
		});
		expect(items.some((i) => i.id === 'manage-skills')).toBe(true);
	});

	it('adds preset apply and web search manage when enabled', () => {
		const items = buildCommandPaletteItems({
			conversations: [],
			presets: [{ id: 'p1', name: 'Coding' }],
			presetsEnabled: true,
			routes,
			skills: [],
			skillsEnabled: false,
			webSearchEnabled: true
		});

		expect(items.find((i) => i.id === 'preset-p1')?.action).toEqual({
			presetId: 'p1',
			type: 'apply-preset'
		});
		expect(items.some((i) => i.id === 'manage-search-providers')).toBe(true);
	});
});

describe('filterCommandPaletteItems / groupCommandPaletteItems', () => {
	const items = buildCommandPaletteItems({
		conversations: [{ id: '1', name: 'Alpha chat' }],
		presets: [],
		presetsEnabled: false,
		routes,
		skills: [{ description: 'lang tools', id: 's', name: 'translate' }],
		skillsEnabled: true,
		webSearchEnabled: false
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
