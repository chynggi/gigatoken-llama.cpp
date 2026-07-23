<script lang="ts">
	import { browser } from '$app/environment';
	import { mode } from 'mode-watcher';

	import githubDarkCss from 'highlight.js/styles/github-dark-dimmed.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';
	import tokyoNightDarkCss from 'highlight.js/styles/tokyo-night-dark.css?inline';
	import tokyoNightLightCss from 'highlight.js/styles/tokyo-night-light.css?inline';
	import nordCss from 'highlight.js/styles/nord.css?inline';
	import draculaCss from 'highlight.js/styles/base16/dracula.css?inline';
	import gruvboxDarkCss from 'highlight.js/styles/base16/gruvbox-dark-medium.css?inline';
	import gruvboxLightCss from 'highlight.js/styles/base16/gruvbox-light-medium.css?inline';
	import grayscaleCss from 'highlight.js/styles/grayscale.css?inline';
	import { ColorMode } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { SYNTAX_CODE_SCROLL_AT_BOTTOM_THRESHOLD_PX } from '$lib/constants/auto-scroll';
	import { highlightCode } from '$lib/utils';

	interface Props {
		code: string;
		language?: string;
		class?: string;
		maxHeight?: string;
		maxWidth?: string;
		/** Auto-scrolls to the bottom of new chunks; pauses on user scroll-up
		 *  until the user returns to the bottom. */
		streaming?: boolean;
	}

	let {
		code,
		language = 'text',
		class: className = '',
		maxHeight = '60vh',
		maxWidth = '',
		streaming = false
	}: Props = $props();

	const highlightedHtml = $derived(highlightCode(code, language));

	let scrollEl = $state<HTMLDivElement>();
	let userScrolledUp = $state(false);
	let lastScrollTop = 0;
	const SCROLL_BOTTOM_THRESHOLD_PX = SYNTAX_CODE_SCROLL_AT_BOTTOM_THRESHOLD_PX;
	let pendingFrame: number | null = null;

	function loadHighlightTheme(isDark: boolean, themeStyle: string) {
		if (!browser) return;

		const existingThemes = document.querySelectorAll('style[data-highlight-theme-preview]');
		existingThemes.forEach((style) => style.remove());

		const style = document.createElement('style');
		style.setAttribute('data-highlight-theme-preview', 'true');
		
		let cssContent = isDark ? githubDarkCss : githubLightCss;

		if (themeStyle === 'tokyo-night') {
			cssContent = isDark ? tokyoNightDarkCss : tokyoNightLightCss;
		} else if (themeStyle === 'nord') {
			cssContent = nordCss;
		} else if (themeStyle === 'dracula' || themeStyle === 'synthwave') {
			cssContent = draculaCss;
		} else if (themeStyle === 'gruvbox') {
			cssContent = isDark ? gruvboxDarkCss : gruvboxLightCss;
		} else if (themeStyle === 'monochrome') {
			cssContent = grayscaleCss;
		}

		style.textContent = cssContent;
		document.head.appendChild(style);
	}

	function isAtBottom(): boolean {
		if (!scrollEl) return false;
		return (
			scrollEl.scrollHeight - scrollEl.clientHeight - scrollEl.scrollTop <=
			SCROLL_BOTTOM_THRESHOLD_PX
		);
	}

	function scrollToBottomOnFrame() {
		if (pendingFrame !== null || !scrollEl || userScrolledUp) return;
		pendingFrame = requestAnimationFrame(() => {
			pendingFrame = null;
			// User may scroll between scheduling and paint.
			if (scrollEl && !userScrolledUp) {
				scrollEl.scrollTop = scrollEl.scrollHeight;
			}
		});
	}

	function handleScrollEvent() {
		if (!scrollEl) return;
		const isScrollingUp = scrollEl.scrollTop < lastScrollTop;
		if (isScrollingUp && !isAtBottom()) {
			userScrolledUp = true;
		} else if (isAtBottom()) {
			userScrolledUp = false;
		}
		lastScrollTop = scrollEl.scrollTop;
	}

	$effect(() => {
		const currentMode = mode.current;
		const isDark = currentMode === ColorMode.DARK;
		const themeStyle = String(config()[SETTINGS_KEYS.THEME_STYLE] ?? 'default');

		loadHighlightTheme(isDark, themeStyle);
	});

	// Pin to bottom at the start of each streaming episode.
	$effect(() => {
		if (streaming) {
			userScrolledUp = false;
			lastScrollTop = 0;
		}
	});

	$effect(() => {
		void code;
		if (!streaming || userScrolledUp) return;
		scrollToBottomOnFrame();
	});

	// Layout shifts that don't change `code` (highlight.js re-tokenize, line-wrap reflow).
	$effect(() => {
		if (!streaming || !scrollEl) return;

		const observer = new MutationObserver(() => scrollToBottomOnFrame());
		observer.observe(scrollEl, {
			childList: true,
			subtree: true,
			characterData: true
		});

		return () => observer.disconnect();
	});
</script>

<div
	bind:this={scrollEl}
	onscroll={handleScrollEvent}
	class="code-preview-wrapper min-w-0 max-w-full overflow-auto rounded-xl border shadow-[0_1px_2px_0_rgb(0_0_0_/_0.05)] {className}"
	style="border-color: color-mix(in oklch, var(--border) 30%, transparent); background: var(--code-background); max-height: {maxHeight}; {maxWidth
		? `max-width: ${maxWidth};`
		: ''}"
>
	<!-- Single line: hljs injection depends on a contiguous source string. -->
	<pre class="m-0"><code class="hljs text-sm leading-relaxed">{@html highlightedHtml}</code></pre>
</div>

<style>
	.code-preview-wrapper {
		overscroll-behavior: contain;
	}

	.code-preview-wrapper pre {
		background: transparent;
		padding: 0;
	}

	.code-preview-wrapper code {
		background: transparent;
		display: block;
		padding: 0.5rem;
	}

	:global(.dark) .code-preview-wrapper {
		border-color: color-mix(in oklch, var(--border) 20%, transparent);
	}
</style>
