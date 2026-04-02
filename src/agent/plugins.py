"""
Plugin loader — scans plugin dir, resolves add_dirs from plugins.yaml.
"""

import os
from pathlib import Path

import yaml
from claude_agent_sdk import SdkPluginConfig


def register_plugins(plugin_dir: str) -> tuple[list[SdkPluginConfig], list[str]]:
    """Scan plugin_dir and return (plugins, add_dirs).

    Args:
        plugin_dir: Directory containing plugin subdirectories and plugins.yaml.

    Returns:
        plugins: SdkPluginConfig list for all accessible plugin subdirs.
        add_dirs: Merged list of plugin dirs, repositories, and extra_dirs.
    """
    base = Path(plugin_dir)

    registry: dict = {}
    yaml_path = base / "plugins.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            registry = yaml.safe_load(f).get("plugins", {})

    plugins: list[SdkPluginConfig] = []
    add_dirs: list[str] = []

    for name in os.listdir(plugin_dir):
        path = base / name
        if not path.is_dir() or name.startswith("."):
            continue
        plugins.append(SdkPluginConfig(type="local", path=str(path)))
        add_dirs.append(str(path))
        cfg = registry.get(name, {})
        add_dirs.extend(str(Path(p)) for p in cfg.get("repositories", []))
        add_dirs.extend(str(Path(p)) for p in cfg.get("extra_dirs", []))

    return plugins, list(dict.fromkeys(add_dirs))  # deduplicate, preserve order
