/**
 * Accent color palette for the fork's theme layer.
 *
 * `ThemeEffects` writes the selected entry onto CSS custom properties, which
 * `fork.css` then consumes.
 */

export interface AccentColorDefinition {
	light: string;
	dark: string;
	glowLight: string;
	glowDark: string;
}

export const ACCENT_COLORS: Record<string, AccentColorDefinition> = {
	default: {
		light: 'oklch(0.93 0.04 220)',
		dark: 'oklch(0.4 0.2 220)',
		glowLight: 'oklch(0.6 0.22 220)',
		glowDark: 'oklch(0.85 0.22 220)'
	},
	blue: {
		light: 'oklch(0.93 0.04 250)',
		dark: 'oklch(0.3 0.14 250)',
		glowLight: 'oklch(0.6 0.18 250)',
		glowDark: 'oklch(0.85 0.14 250)'
	},
	green: {
		light: 'oklch(0.94 0.04 145)',
		dark: 'oklch(0.32 0.14 145)',
		glowLight: 'oklch(0.65 0.18 145)',
		glowDark: 'oklch(0.85 0.14 145)'
	},
	purple: {
		light: 'oklch(0.92 0.05 290)',
		dark: 'oklch(0.3 0.15 290)',
		glowLight: 'oklch(0.58 0.2 290)',
		glowDark: 'oklch(0.82 0.15 290)'
	},
	orange: {
		light: 'oklch(0.93 0.05 55)',
		dark: 'oklch(0.32 0.14 55)',
		glowLight: 'oklch(0.65 0.18 55)',
		glowDark: 'oklch(0.85 0.14 55)'
	},
	pink: {
		light: 'oklch(0.93 0.05 350)',
		dark: 'oklch(0.32 0.14 350)',
		glowLight: 'oklch(0.65 0.18 350)',
		glowDark: 'oklch(0.85 0.14 350)'
	},
	red: {
		light: 'oklch(0.92 0.05 25)',
		dark: 'oklch(0.3 0.14 25)',
		glowLight: 'oklch(0.6 0.18 25)',
		glowDark: 'oklch(0.82 0.14 25)'
	}
};
