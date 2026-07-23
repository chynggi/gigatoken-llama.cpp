/**
 * presetsStore - Reactive store for chat presets.
 *
 * A preset is a saved combination of settings: system message, sampling
 * parameters, MCP overrides, and web search preferences.
 */

import { browser } from '$app/environment';
import { PresetService } from '$lib/services/preset.service';
import { settingsStore, config } from '$lib/stores/settings.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { buildPresetConfigUpdates } from '$lib/utils/preset-apply';

class PresetsStore {
	presets = $state<DatabasePreset[]>([]);
	isInitialized = $state(false);

	get enabled(): boolean {
		return Boolean(config()[SETTINGS_KEYS.PRESETS_ENABLED]);
	}

	async init(): Promise<void> {
		if (!browser) return;
		if (this.isInitialized) return;
		try {
			this.presets = await PresetService.getAll();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize presets:', error);
		}
	}

	async refresh(): Promise<void> {
		this.presets = await PresetService.getAll();
	}

	async createPreset(data: Omit<DatabasePreset, 'id' | 'createdAt'>): Promise<DatabasePreset> {
		const preset = await PresetService.create(data);
		this.presets = [...this.presets, preset];
		return preset;
	}

	async updatePreset(id: string, updates: Partial<DatabasePreset>): Promise<void> {
		await PresetService.update(id, updates);
		const idx = this.presets.findIndex((p) => p.id === id);
		if (idx !== -1) {
			this.presets[idx] = { ...this.presets[idx], ...updates };
			this.presets = [...this.presets];
		}
	}

	async deletePreset(id: string): Promise<void> {
		await PresetService.delete(id);
		this.presets = this.presets.filter((p) => p.id !== id);
	}

	/**
	 * Apply a preset - set all its config values on the current settings.
	 * Field mapping lives in buildPresetConfigUpdates (pure, unit-tested).
	 * Reloads pending MCP defaults so new chats pick up mcpOverrides without remount.
	 */
	applyPreset(presetId: string): void {
		const preset = this.presets.find((p) => p.id === presetId);
		if (!preset) return;

		let touchedMcp = false;
		for (const { key, value } of buildPresetConfigUpdates(preset)) {
			settingsStore.updateConfig(key, value);
			if (key === SETTINGS_KEYS.MCP_DEFAULT_SERVER_OVERRIDES) {
				touchedMcp = true;
			}
		}

		// New chats use pendingMcpServerOverrides, not settings alone
		if (touchedMcp || preset.mcpOverrides !== undefined) {
			conversationsStore.reloadPendingMcpFromSettings();
		}
	}
}

export const presetsStore = new PresetsStore();

if (browser) {
	presetsStore.init();
}
