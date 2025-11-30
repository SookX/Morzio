package com.morzio.pos

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.navigation.compose.rememberNavController
import com.morzio.pos.navigation.NavGraph
import com.morzio.pos.ui.theme.POSTheme
import recieptservice.com.recieptservice.PrinterInterface
import java.util.concurrent.atomic.AtomicReference

val LocalPrinterService = staticCompositionLocalOf<PrinterServiceAccessor?> { null }

interface PrinterServiceAccessor {
    fun getPrinterService(): PrinterInterface?
    fun isBound(): Boolean
    fun attemptBind()
}

class MainActivity : ComponentActivity(), PrinterServiceAccessor {
    private val printerServiceRef = AtomicReference<PrinterInterface?>()
    @Volatile private var isPrinterServiceBound = false

    private val printerServiceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            Log.i("MainActivity", "Printer Service Connected")
            printerServiceRef.set(PrinterInterface.Stub.asInterface(service))
            isPrinterServiceBound = true
            Toast.makeText(this@MainActivity, "Printer Ready", Toast.LENGTH_SHORT).show()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.w("MainActivity", "Printer Service Disconnected")
            printerServiceRef.set(null)
            isPrinterServiceBound = false
            Toast.makeText(this@MainActivity, "Printer Disconnected", Toast.LENGTH_SHORT).show()
        }

        override fun onBindingDied(name: ComponentName?) {
            Log.e("MainActivity", "Printer Service Binding Died")
            onServiceDisconnected(name)
        }

        override fun onNullBinding(name: ComponentName?) {
            Log.e("MainActivity", "Printer Service Null Binding")
            isPrinterServiceBound = false
            Toast.makeText(this@MainActivity, "Printer Binding Failed", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        bindToPrinterService()
        
        setContent {
            CompositionLocalProvider(LocalPrinterService provides this) {
                POSTheme(darkTheme = true) {
                    Surface(
                        modifier = Modifier.fillMaxSize(),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        val navController = rememberNavController()
                        NavGraph(navController = navController)
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unbindPrinterService()
    }

    override fun getPrinterService(): PrinterInterface? {
        return if (isPrinterServiceBound) printerServiceRef.get() else null
    }

    override fun isBound(): Boolean {
        return isPrinterServiceBound
    }

    override fun attemptBind() {
        bindToPrinterService()
    }

    private fun bindToPrinterService() {
        if (!isPrinterServiceBound) {
            val intent = Intent()
            val servicePackage = "recieptservice.com.recieptservice"
            val serviceClass = "recieptservice.com.recieptservice.service.PrinterService"
            intent.component = ComponentName(servicePackage, serviceClass)

            try {
                val bound = bindService(intent, printerServiceConnection, Context.BIND_AUTO_CREATE)
                if (!bound) {
                    Log.e("MainActivity", "bindService returned false")
                    Toast.makeText(this, "Printer Service Not Found", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Exception binding printer service", e)
            }
        }
    }

    private fun unbindPrinterService() {
        if (isPrinterServiceBound) {
            try {
                unbindService(printerServiceConnection)
            } catch (e: Exception) {
                Log.w("MainActivity", "Service already unbound", e)
            } finally {
                isPrinterServiceBound = false
                printerServiceRef.set(null)
            }
        }
    }
}