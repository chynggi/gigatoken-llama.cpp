import { skillsTable } from '$lib/fork/db/tables';
import type { DatabaseSkill } from '$lib/fork/db/types';
import { BUILT_IN_SKILLS } from '$lib/fork/packs/built-in-skills';
import { uuid } from '$lib/utils';

export class SkillService {
	static async getAll(): Promise<DatabaseSkill[]> {
		return await skillsTable.orderBy('createdAt').toArray();
	}

	static async get(id: string): Promise<DatabaseSkill | undefined> {
		return await skillsTable.get(id);
	}

	static async getByName(name: string): Promise<DatabaseSkill | undefined> {
		return await skillsTable.where('name').equals(name).first();
	}

	static async search(query: string): Promise<DatabaseSkill[]> {
		const lower = query.toLowerCase();
		return await skillsTable
			.filter(
				(s) => s.name.toLowerCase().includes(lower) || s.description.toLowerCase().includes(lower)
			)
			.toArray();
	}

	static async create(skill: Omit<DatabaseSkill, 'id' | 'createdAt'>): Promise<DatabaseSkill> {
		const newSkill: DatabaseSkill = {
			...skill,
			id: uuid(),
			createdAt: Date.now()
		};
		await skillsTable.add(newSkill);
		return newSkill;
	}

	static async update(id: string, updates: Partial<Omit<DatabaseSkill, 'id'>>): Promise<void> {
		await skillsTable.update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		const skill = await skillsTable.get(id);
		if (skill?.isBuiltIn) {
			throw new Error('Built-in skills cannot be deleted');
		}
		await skillsTable.delete(id);
	}

	static async recordUsage(id: string): Promise<void> {
		const skill = await skillsTable.get(id);
		if (!skill) return;
		await skillsTable.update(id, {
			lastUsedAt: Date.now(),
			usageCount: (skill.usageCount ?? 0) + 1
		});
	}

	static async seedBuiltInSkills(): Promise<void> {
		const existing = await skillsTable.filter((s) => s.isBuiltIn === true).count();
		if (existing > 0) return;

		for (const skill of BUILT_IN_SKILLS) {
			await SkillService.create(skill);
		}
	}
}
