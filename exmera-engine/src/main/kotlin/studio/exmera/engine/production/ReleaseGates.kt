package studio.exmera.engine.production

/** Release pipeline stages. Gate/report/evaluator live in ProductionHardening.kt (single source of truth). */
enum class ReleaseStage{INTERNAL,BETA,RELEASE}

object HardeningDefaults{const val MAX_EXPORT_PIXELS:Long=40_000_000;const val MAX_PROJECT_CLIPS=1000;const val MAX_EFFECTS_PER_FRAME=64;const val MAX_MODEL_FILE_BYTES:Long=2L*1024*1024*1024}
