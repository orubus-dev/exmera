package app.orubus.exmera.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.orubus.exmera.packs.ExmPackManager
import android.net.Uri

/** Model packs sheet — the EXM runtime entry point. The client ships with zero models. */
@Composable
fun PacksSheet(onDismiss: () -> Unit, onStatus: (String) -> Unit) {
    val context = LocalContext.current
    val manager = remember { ExmPackManager(ExmPackManager.defaultRoot(context)) }
    var packs by remember { mutableStateOf(manager.list()) }
    var message by remember { mutableStateOf("no packs installed — exmera ships clean") }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val input = context.contentResolver.openInputStream(uri)
                ?: throw IllegalStateException("cannot open selected file")
            val name = uri.lastPathSegment?.substringAfterLast('/') ?: "pack.exm"
            val info = manager.importFrom(input, name)
            packs = manager.list()
            message = "verified · ${info.name} · chunks OK · ${ExmPackManager.formatBytes(info.sizeBytes)}"
            onStatus("pack installed: ${info.name}")
        } catch (t: Throwable) {
            message = "import failed: ${t.message}"
        }
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(Mono.glassStrong)
            .pointerInput(onDismiss) { detectTapGestures { onDismiss() } },
        contentAlignment = Alignment.Center
    ) {
        GlassPanel(
            Modifier
                .padding(20.dp)
                .fillMaxWidth()
                .height(520.dp)
                .pointerInput(Unit) { detectTapGestures { } },
            corner = 28.dp,
            strong = true
        ) {
            Column(Modifier.padding(18.dp).verticalScroll(rememberScrollState())) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("model packs", color = Mono.ink, fontSize = 15.sp, modifier = Modifier.weight(1f))
                    MonoIcon("close", Modifier.size(20.dp).pointerInput(onDismiss) { detectTapGestures { onDismiss() } })
                }
                Text(
                    "exmera ships with zero models. everything learned arrives as a .exm pack — header-verified and chunk-checked before install.",
                    color = Mono.inkDim, fontSize = 11.sp, modifier = Modifier.padding(top = 6.dp)
                )
                Spacer(Modifier.height(12.dp))
                GlassChip("import .exm", selected = true) {
                    picker.launch(arrayOf("*/*"))
                }
                Spacer(Modifier.height(6.dp))
                Text(message, color = Mono.inkFaint, fontSize = 10.sp)
                Spacer(Modifier.height(12.dp))
                if (packs.isEmpty()) {
                    Text(
                        "installed: none",
                        color = Mono.inkDim, fontSize = 12.sp
                    )
                } else {
                    packs.forEach { p ->
                        GlassPanel(Modifier.fillMaxWidth().padding(vertical = 4.dp), corner = 18.dp) {
                            Column(Modifier.padding(12.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Column(Modifier.weight(1f)) {
                                        Text("${p.name} · ${p.version}", color = Mono.ink, fontSize = 12.sp)
                                        Text(p.fileName, color = Mono.inkFaint, fontSize = 9.sp)
                                    }
                                    MonoIcon(
                                        "close",
                                        Modifier.size(16.dp).pointerInput(p.fileName) {
                                            detectTapGestures {
                                                manager.remove(p.fileName)
                                                packs = manager.list()
                                                message = "removed ${p.name}"
                                            }
                                        },
                                        tint = Mono.inkDim
                                    )
                                }
                                Text(
                                    ExmPackManager.summarize(p),
                                    color = Mono.inkDim, fontSize = 10.sp, modifier = Modifier.padding(top = 4.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/** Privacy sheet — states exactly what exmera writes and strips. */
@Composable
fun PrivacySheet(onDismiss: () -> Unit) {
    Box(
        Modifier
            .fillMaxSize()
            .background(Mono.glassStrong)
            .pointerInput(onDismiss) { detectTapGestures { onDismiss() } },
        contentAlignment = Alignment.Center
    ) {
        GlassPanel(
            Modifier
                .padding(20.dp)
                .fillMaxWidth()
                .pointerInput(Unit) { detectTapGestures { } },
            corner = 28.dp,
            strong = true
        ) {
            Column(Modifier.padding(18.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("metadata privacy", color = Mono.ink, fontSize = 15.sp, modifier = Modifier.weight(1f))
                    MonoIcon("close", Modifier.size(20.dp).pointerInput(onDismiss) { detectTapGestures { onDismiss() } })
                }
                Spacer(Modifier.height(10.dp))
                PrivacyLine("every jpeg is a fresh pixel-only re-encode — exif, gps, maker notes, thumbnails and icc payloads never reach the gallery")
                PrivacyLine("rotation is baked into pixels so scrubbed files display correctly without orientation tags")
                PrivacyLine("raw dng keeps only its technical tags (a raw negative without them is corrupt) — and never receives gps")
                PrivacyLine("video files are written without any location metadata, ever")
                PrivacyLine("no analytics, no telemetry, no network permission")
            }
        }
    }
}

@Composable
private fun PrivacyLine(text: String) {
    Row(Modifier.padding(vertical = 5.dp)) {
        Text("·", color = Mono.inkFaint, fontSize = 11.sp, modifier = Modifier.width(14.dp))
        Text(text, color = Mono.inkDim, fontSize = 11.sp)
    }
}
