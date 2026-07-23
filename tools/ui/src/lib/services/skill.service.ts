import { BUILT_IN_SKILLS, IDXDB_TABLES } from '$lib/constants';
import { uuid } from '$lib/utils';
import { db } from './database.service';

export class SkillService {
	static async getAll(): Promise<DatabaseSkill[]> {
		return await db[IDXDB_TABLES.skills].orderBy('createdAt').toArray();
	}

	static async get(id: string): Promise<DatabaseSkill | undefined> {
		return await db[IDXDB_TABLES.skills].get(id);
	}

	static async getByName(name: string): Promise<DatabaseSkill | undefined> {
		return await db[IDXDB_TABLES.skills].where('name').equals(name).first();
	}

	static async search(query: string): Promise<DatabaseSkill[]> {
		const lower = query.toLowerCase();
		return await db[IDXDB_TABLES.skills]
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
		await db[IDXDB_TABLES.skills].add(newSkill);
		return newSkill;
	}

	static async update(id: string, updates: Partial<Omit<DatabaseSkill, 'id'>>): Promise<void> {
		await db[IDXDB_TABLES.skills].update(id, updates);
	}

	static async delete(id: string): Promise<void> {
		const skill = await db[IDXDB_TABLES.skills].get(id);
		if (skill?.isBuiltIn) {
			throw new Error('Built-in skills cannot be deleted');
		}
		await db[IDXDB_TABLES.skills].delete(id);
	}

	static async recordUsage(id: string): Promise<void> {
		const skill = await db[IDXDB_TABLES.skills].get(id);
		if (!skill) return;
		await db[IDXDB_TABLES.skills].update(id, {
			lastUsedAt: Date.now(),
			usageCount: (skill.usageCount ?? 0) + 1
		});
	}

	static async seedBuiltInSkills(): Promise<void> {
		const existing = await db[IDXDB_TABLES.skills].filter((s) => s.isBuiltIn === true).count();
		if (existing > 0) return;

		for (const skill of BUILT_IN_SKILLS) {
			await SkillService.create(skill);
		}
	}
}
