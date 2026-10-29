package app.orubus.exmera

import android.Manifest
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.orubus.exmera.ui.ExmeraApp
import app.orubus.exmera.ui.GlassChip
import app.orubus.exmera.ui.GlassPanel
import app.orubus.exmera.ui.Mono
import studio.exmera.engine.ExmeraEngine

class MainActivity : ComponentActivity() {

    private val granted = mutableStateOf(false)
    private lateinit var permissionLauncher: androidx.activity.result.ActivityResultLauncher<Array<String>>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { result -> granted.value = result.values.all { it } }

        val alreadyGranted = androidx.core.content.ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        if (alreadyGranted) granted.value = true

        setContent {
            if (granted.value) {
                ExmeraApp(profileName())
            } else {
                PermissionGate { permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)) }
            }
        }
    }

    private fun profileName(): String = runCatching {
        ExmeraEngine(this).profileName()
    }.getOrDefault("camera")
}

@Composable
private fun PermissionGate(onRequest: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
        GlassPanel(Modifier.padding(28.dp)) {
            Column(
                Modifier.padding(horizontal = 26.dp, vertical = 22.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("exmera", color = Mono.ink, fontSize = 17.sp)
                Spacer(Modifier.height(8.dp))
                Text(
                    "the camera is the instrument.\ngrant camera access to begin.",
                    color = Mono.inkDim, fontSize = 12.sp, lineHeight = 18.sp
                )
                Spacer(Modifier.height(16.dp))
                GlassChip("grant camera", selected = true) { onRequest() }
            }
        }
    }
}
