export const BOX_BORDER =
	'border border-border/30 focus-within:border-border  dark:border-border/20 dark:focus-within:border-border';

export const INPUT_CLASSES = `
    bg-background/75 dark:bg-background/60
    ${BOX_BORDER}
    shadow-sm
    outline-none
    text-foreground
    glass-backdrop
    transition-shadow duration-200
    glass-focus-ring
`;

export const PANEL_CLASSES = `
    bg-background/85 dark:bg-background/70
    border border-border/30 dark:border-border/20
    shadow-sm backdrop-blur-xl!
    rounded-t-lg!
`;

export const CHAT_FORM_POPOVER_MAX_HEIGHT = 'max-h-80';
export const DIALOG_SUBMENU_CONTENT = 'w-60';

/** Default Tailwind size class for inline icon components (lucide, etc.). */
export const ICON_CLASS_DEFAULT = 'h-4 w-4';

/** Icon size + spinning animation; used for live-streaming tool indicators. */
export const ICON_CLASS_SPIN = 'h-4 w-4 animate-spin';
