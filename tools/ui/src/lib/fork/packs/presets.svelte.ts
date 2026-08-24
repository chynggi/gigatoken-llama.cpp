/**
 * presetsStore - Reactive store for chat presets.
 *
 * A preset is a saved combination of settings: system message, sampling
 * parameters, MCP overrides, and web search preferences.
 */

import { browser } from '$app/environment';
import { SETTINGS_KEYS } from '$lib/constants';
import type { DatabasePreset } from '$lib/fork/db/types';
import { buildPresetConfigUpdates } from '$lib/fork/packs/preset-apply';
import { PresetService } from '$lib/fork/packs/preset.service';
import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
import { settingsStore } from '$lib/stores';

class PresetsStore {
	presets = $state<DatabasePreset[]>([]);
	isInitialized = $state(false);

	get enabled(): boolean {
		return Boolean(settingsStore.config[FORK_SETTINGS_KEYS.PRESETS_ENABLED]);
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
	 *
	 * MCP overrides are merged onto `mcpServers[i].enabled`, which upstream
	 * treats as the single source of truth for new-chat defaults, so no
	 * separate reload step is needed for new chats to pick them up.
	 */
	applyPreset(presetId: string): void {
		const preset = this.presets.find((p) => p.id === presetId);
		if (!preset) return;

		const currentMcpServers = settingsStore.config[SETTINGS_KEYS.MCP_SERVERS];

		for (const { key, value } of buildPresetConfigUpdates(
			preset,
			typeof currentMcpServers === 'string' ? currentMcpServers : undefined
		)) {
			settingsStore.updateConfig(key, value);
		}
	}
}

export const presetsStore = new PresetsStore();

if (browser) {
	presetsStore.init();
}
