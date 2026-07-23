/**
 * Skill engine - parse slash commands and apply skill templates.
 *
 * Parses input like "/translate lang=ko text=Hello world" into a
 * skill name and key=value arguments, then substitutes them into
 * the skill's content template.
 */

import type { DatabaseSkill, SkillPlaceholder } from '$lib/types/database';

export interface ParsedSkillCommand {
	skillName: string;
	rawArgs: string;
	args: Record<string, string>;
}

/**
 * Parse a slash-command string into skill name and key=value arguments.
 *
 * Examples:
 *   "/summarize"            => { skillName: "summarize", args: {} }
 *   "/translate lang=ko"    => { skillName: "translate", args: { lang: "ko" } }
 *   "/email topic=Hello tone=formal" => { skillName: "email", args: { topic: "Hello", tone: "formal" } }
 */
export function parseSkillCommand(input: string): ParsedSkillCommand | null {
	const trimmed = input.trim();
	if (!trimmed.startsWith('/')) return null;

	const rest = trimmed.slice(1);
	if (rest.length === 0) return null;

	const spaceIdx = rest.indexOf(' ');
	const skillName = spaceIdx === -1 ? rest : rest.slice(0, spaceIdx);
	if (skillName.length === 0) return null;

	const rawArgs = spaceIdx === -1 ? '' : rest.slice(spaceIdx + 1);
	const args = parseSkillArgs(rawArgs);

	return { skillName, rawArgs, args };
}

/**
 * Parse "key=value key2=value2 remaining free text" into a map.
 *
 * Key=value pairs are parsed greedily: the value extends until the next
 * key= pattern or end of string. Trailing free text (text not part of
 * any key=value pair) is stored under the "text" key.
 */
function parseSkillArgs(raw: string): Record<string, string> {
	const result: Record<string, string> = {};
	const trimmed = raw.trim();
	if (!trimmed) return result;

	// Greedy regex: match key= then capture value until next key= or end
	const keyValRe = /([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*/g;
	const keys: { key: string; start: number; eqEnd: number }[] = [];

	let m: RegExpExecArray | null;
	while ((m = keyValRe.exec(trimmed)) !== null) {
		keys.push({ key: m[1], start: m.index, eqEnd: m.index + m[0].length });
	}

	if (keys.length === 0) {
		result['text'] = trimmed;
		return result;
	}

	// Any text before the first key= is trailing text
	if (keys[0].start > 0) {
		const prefix = trimmed.slice(0, keys[0].start).trim();
		if (prefix) result['text'] = prefix;
	}

	for (let i = 0; i < keys.length; i++) {
		const { key, eqEnd } = keys[i];
		const valueEnd = i + 1 < keys.length ? keys[i + 1].start : trimmed.length;
		let value = trimmed.slice(eqEnd, valueEnd).trim();
		// Strip surrounding quotes (single or double)
		if (
			value.length >= 2 &&
			((value.startsWith('"') && value.endsWith('"')) ||
				(value.startsWith("'") && value.endsWith("'")))
		) {
			value = value.slice(1, -1);
		}
		result[key] = value;
	}

	return result;
}

/**
 * Apply a skill template by substituting {{placeholder}} variables with
 * provided arguments. Uses default values from skill.placeholders when
 * an argument is missing.
 */
export function applySkillTemplate(
	skill: DatabaseSkill,
	args: Record<string, string>
): string {
	let result = skill.content;

	if (skill.placeholders && skill.placeholders.length > 0) {
		for (const ph of skill.placeholders) {
			const value = args[ph.name] ?? ph.defaultValue ?? '';
			result = result.replaceAll(`{{${ph.name}}}`, value);
		}

		// Replace any remaining unmatched placeholders with empty string
		result = result.replace(/\{\{(\w+)\}\}/g, '');

		return result;
	}

	// Simple case: no placeholders defined, just substitute named args
	for (const [key, value] of Object.entries(args)) {
		result = result.replaceAll(`{{${key}}}`, value);
	}
	result = result.replace(/\{\{(\w+)\}\}/g, '');

	return result;
}

/**
 * Get placeholder values needed by a skill that are not yet filled.
 * Used by the UI to show which arguments are still required.
 */
export function missingPlaceholders(
	skill: DatabaseSkill,
	args: Record<string, string>
): SkillPlaceholder[] {
	if (!skill.placeholders) return [];
	return skill.placeholders.filter(
		(ph) => !(ph.name in args) && ph.defaultValue === undefined
	);
}

/**
 * Find skills matching a partial name. Returns results ranked by:
 * 1. Exact name match
 * 2. Name starts with query
 * 3. Name contains query
 * 4. Description contains query
 * Within each tier, sort by usageCount descending then by lastUsedAt.
 */
export function rankSkills(
	skills: DatabaseSkill[],
	query: string
): DatabaseSkill[] {
	const lower = query.toLowerCase();
	if (!lower) return skills;

	const scored = skills.map((s) => {
		const nameLower = s.name.toLowerCase();
		let score = 0;

		if (nameLower === lower) {
			score = 4;
		} else if (nameLower.startsWith(lower)) {
			score = 3;
		} else if (nameLower.includes(lower)) {
			score = 2;
		} else if (s.description.toLowerCase().includes(lower)) {
			score = 1;
		}

		return { skill: s, score };
	});

	return scored
		.filter(({ score }) => score > 0)
		.sort((a, b) => {
			if (b.score !== a.score) return b.score - a.score;
			const usageA = a.skill.usageCount ?? 0;
			const usageB = b.skill.usageCount ?? 0;
			if (usageB !== usageA) return usageB - usageA;
			return (b.skill.lastUsedAt ?? 0) - (a.skill.lastUsedAt ?? 0);
		})
		.map(({ skill }) => skill);
}
