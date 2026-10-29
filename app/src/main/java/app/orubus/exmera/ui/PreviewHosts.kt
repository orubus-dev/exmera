package app.orubus.exmera.ui

import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.view.PreviewView
import android.view.TextureView
import androidx.lifecycle.compose.LocalLifecycleOwner
import app.orubus.exmera.camera.CameraMode
import app.orubus.exmera.camera.ExmeraCameraController

/**
 * CameraX preview host for IMAGE / VIDEO / PORTRAIT / PANO modes.
 * Rebinds use cases when the mode changes or the user flips the camera.
 */
@Composable
fun CameraXPreviewHost(
    holder: ExmeraCameraControllerHolder,
    mode: CameraMode,
    modifier: Modifier,
    flipToken: Int,
    onTap: (Float, Float) -> Unit
) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val controller = remember { holder.getOrCreate(context, owner) }
    var view by remember { mutableStateOf<PreviewView?>(null) }

    AndroidView(
        factory = { ctx -> PreviewView(ctx).also { view = it } },
        modifier = modifier.pointerInput(Unit) {
            detectTapGestures { offset ->
                onTap(offset.x / size.width.coerceAtLeast(1), offset.y / size.height.coerceAtLeast(1))
            }
        }
    )

    LaunchedEffect(view, mode, flipToken) {
        val v = view ?: return@LaunchedEffect
        controller.bind(v, mode)
    }

    DisposableEffect(Unit) {
        onDispose { controller.pause() }
    }
}

/** Native Camera2 preview host for PRO mode. Fresh stack per entry (thread safety). */
@Composable
fun ProPreviewHost(holder: ProCamera2Holder, modifier: Modifier) {
    val context = LocalContext.current
    AndroidView(
        factory = { ctx ->
            val pro = holder.getOrCreate(ctx)
            TextureView(ctx).also { pro.open(it) }
        },
        modifier = modifier
    )
    DisposableEffect(Unit) {
        onDispose { holder.dispose() }
    }
}
