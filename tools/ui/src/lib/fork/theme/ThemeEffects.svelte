<script lang="ts">
	import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
	import { ACCENT_COLORS } from '$lib/fork/theme/accent-colors';
	import { settingsStore } from '$lib/stores';

	const WIDE_CHAT_CLASS = 'wide-chat-mode';
	const FULL_CHAT_CLASS = 'full-chat-mode';

	const THEME_CLASSES = [
		'theme-tokyo-night',
		'theme-nord',
		'theme-dracula',
		'theme-gruvbox',
		'theme-synthwave',
		'theme-soft',
		'theme-monochrome'
	];

	$effect(() => {
		const currentConfig = settingsStore.config;
		const accentKey = String(currentConfig[FORK_SETTINGS_KEYS.ACCENT_COLOR] ?? 'default');
		const accent = ACCENT_COLORS[accentKey] ?? ACCENT_COLORS.default;
		const themeStyle = String(currentConfig[FORK_SETTINGS_KEYS.THEME_STYLE] ?? 'default');
		const root = document.documentElement;

		// Theme style CSS owns palette + glow. Only pin accent/glow inline when the user
		// picks a non-default accent, or when no theme style is active (default palette).
		const pinAccentInline = accentKey !== 'default' || themeStyle === 'default';

		if (pinAccentInline) {
			root.style.setProperty('--accent-light', accent.light);
			root.style.setProperty('--accent-dark', accent.dark);
			root.style.setProperty('--glow-color-light', accent.glowLight);
			root.style.setProperty('--glow-color-dark', accent.glowDark);
		} else {
			root.style.removeProperty('--accent-light');
			root.style.removeProperty('--accent-dark');
			root.style.removeProperty('--glow-color-light');
			root.style.removeProperty('--glow-color-dark');
		}

		THEME_CLASSES.forEach((cls) => {
			root.classList.remove(cls);
		});

		if (themeStyle !== 'default') {
			root.classList.add(`theme-${themeStyle}`);
		}

		const chatWidthStyle = String(currentConfig[FORK_SETTINGS_KEYS.CHAT_WIDTH_STYLE] ?? 'normal');

		root.classList.remove(WIDE_CHAT_CLASS);
		root.classList.remove(FULL_CHAT_CLASS);

		if (chatWidthStyle === 'wide') {
			root.classList.add(WIDE_CHAT_CLASS);
		} else if (chatWidthStyle === 'full') {
			root.classList.add(FULL_CHAT_CLASS);
		}
	});
</script>
