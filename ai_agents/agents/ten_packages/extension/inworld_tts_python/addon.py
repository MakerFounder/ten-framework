from ten_runtime import (
    Addon,
    register_addon_as_extension,
    TenEnv,
)


@register_addon_as_extension("inworld_tts_python")
class InworldTTSExtensionAddon(Addon):
    def on_create_instance(self, ten_env: TenEnv, name: str, context) -> None:
        from .extension import InworldTTSExtension

        ten_env.log_info("InworldTTSExtensionAddon on_create_instance")
        ten_env.on_create_instance_done(InworldTTSExtension(name), context)
