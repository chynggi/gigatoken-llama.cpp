/**
 * Routes owned by the fork.
 *
 * Kept out of upstream's `ROUTES` so `routes.constants.ts` stays untouched.
 * These paths back the pages under `src/routes/{skills,presets,search-providers}`.
 */

export const FORK_ROUTES = {
	PRESETS: '#/presets',
	SEARCH_PROVIDERS: '#/search-providers',
	SKILLS: '#/skills'
} as const;
