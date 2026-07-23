/**
 * skillsStore - Reactive store for the skill system.
 *
 * Wraps SkillService with Svelte 5 reactivity. Manages skill list state,
 * search, usage tracking, and built-in skill seeding.
 */

import { browser } from '$app/environment';
import { SkillService } from '$lib/services/skill.service';
import {
	rankSkills,
	parseSkillCommand,
	applySkillTemplate,
	missingPlaceholders
} from '$lib/utils/skill-engine';
import { config } from '$lib/stores/settings.svelte';
import { SETTINGS_KEYS } from '$lib/constants';

class SkillsStore {
	skills = $state<DatabaseSkill[]>([]);
	isInitialized = $state(false);

	get enabled(): boolean {
		return Boolean(config()[SETTINGS_KEYS.SKILL_SYSTEM_ENABLED]);
	}

	async init(): Promise<void> {
		if (!browser) return;
		if (this.isInitialized) return;
		try {
			await SkillService.seedBuiltInSkills();
			await this.loadSkills();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize skills store:', error);
		}
	}

	async loadSkills(): Promise<void> {
		this.skills = await SkillService.getAll();
	}

	/** Search skills by name/description and rank by relevance */
	search(query: string): DatabaseSkill[] {
		return rankSkills(this.skills, query);
	}

	/** Find a skill by exact name */
	findByName(name: string): DatabaseSkill | undefined {
		return this.skills.find((s) => s.name === name);
	}

	/**
	 * Parse input text for a skill command.
	 * - expanded: template filled and ready to send
	 * - incomplete: skill matched but required placeholders missing
	 * - null: not a skill command (or skills disabled)
	 */
	resolveSkillCommand(
		input: string
	):
		| { status: 'expanded'; expanded: string; skill: DatabaseSkill }
		| { status: 'incomplete'; skill: DatabaseSkill; missing: string[] }
		| null {
		if (!this.enabled) return null;
		const parsed = parseSkillCommand(input);
		if (!parsed) return null;

		const skill = this.findByName(parsed.skillName);
		if (!skill) return null;

		const missing = missingPlaceholders(skill, parsed.args);
		if (missing.length > 0) {
			return { status: 'incomplete', skill, missing: missing.map((p) => p.name) };
		}

		const expanded = applySkillTemplate(skill, parsed.args);
		SkillService.recordUsage(skill.id);
		return { status: 'expanded', expanded, skill };
	}

	async createSkill(data: Omit<DatabaseSkill, 'id' | 'createdAt'>): Promise<DatabaseSkill> {
		const skill = await SkillService.create(data);
		this.skills = [...this.skills, skill];
		return skill;
	}

	async updateSkill(id: string, updates: Partial<Omit<DatabaseSkill, 'id'>>): Promise<void> {
		await SkillService.update(id, updates);
		const idx = this.skills.findIndex((s) => s.id === id);
		if (idx !== -1) {
			this.skills[idx] = { ...this.skills[idx], ...updates };
			this.skills = [...this.skills];
		}
	}

	async deleteSkill(id: string): Promise<void> {
		await SkillService.delete(id);
		this.skills = this.skills.filter((s) => s.id !== id);
	}
}

export const skillsStore = new SkillsStore();

if (browser) {
	skillsStore.init();
}
