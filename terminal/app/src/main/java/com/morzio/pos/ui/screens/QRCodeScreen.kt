package com.morzio.pos.ui.screens

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.QrCodeScanner
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.morzio.pos.models.PaymentStatus
import com.morzio.pos.viewmodels.QRCodeViewModel
import com.morzio.pos.LocalPrinterService
import com.morzio.pos.utils.PrinterHelper

@Composable
fun QRCodeScreen(
    amount: String?,
    onNavigateToSuccess: () -> Unit,
    onNavigateToError: (String) -> Unit,
    onCancel: () -> Unit
) {
    val viewModel: QRCodeViewModel = viewModel()
    val uiState by viewModel.uiState.collectAsState()
    val printerAccessor = LocalPrinterService.current
    var showCancelDialog by remember { mutableStateOf(false) }

    LaunchedEffect(amount) {
        amount?.let { viewModel.initialize(it) }
    }

    LaunchedEffect(uiState.status) {
        when (uiState.status) {
            PaymentStatus.COMPLETED -> {
                printerAccessor?.getPrinterService()?.let { service ->
                    PrinterHelper.printReceipt(
                        printerService = service,
                        amount = uiState.amount,
                        sessionId = uiState.sessionId,
                        status = "COMPLETED",
                        paymentUrl = uiState.paymentUrl,
                        installments = uiState.installments
                    )
                }
                onNavigateToSuccess()
            }
            PaymentStatus.TIMEOUT -> onNavigateToError("Transaction timed out")
            PaymentStatus.ERROR -> onNavigateToError("Transaction cancelled")
            else -> {}
        }
    }

    if (showCancelDialog) {
        AlertDialog(
            onDismissRequest = { showCancelDialog = false },
            title = { Text("Cancel Transaction") },
            text = { Text("Are you sure you want to cancel this transaction?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        showCancelDialog = false
                        viewModel.cancelTransaction()
                    }
                ) {
                    Text("Yes, Cancel")
                }
            },
            dismissButton = {
                TextButton(onClick = { showCancelDialog = false }) {
                    Text("No, Continue")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Morzio Pay",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                )
                IconButton(onClick = { showCancelDialog = true }) {
                    Icon(Icons.Default.Close, contentDescription = "Cancel")
                }
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.SpaceEvenly
        ) {
            PaymentInfoCard(amount = uiState.amount, sessionId = uiState.sessionId)
            
            QRCodeCard(bitmap = uiState.qrCodeBitmap)
            
            StatusSection(status = uiState.status)
        }
    }
}

@Composable
fun PaymentInfoCard(amount: Double, sessionId: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = "Total Amount",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = "€${String.format("%.2f", amount)}",
            style = MaterialTheme.typography.displayMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(modifier = Modifier.height(8.dp))
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = RoundedCornerShape(8.dp)
        ) {
            Text(
                text = "ID: ${sessionId.take(8)}...",
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun QRCodeCard(bitmap: android.graphics.Bitmap?) {
    Card(
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Box(
            modifier = Modifier
                .size(320.dp)
                .padding(24.dp),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "QR Code",
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                CircularProgressIndicator(
                    modifier = Modifier.size(48.dp),
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
    
    Spacer(modifier = Modifier.height(16.dp))
    
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
        modifier = Modifier.fillMaxWidth()
    ) {
        Icon(
            imageVector = Icons.Default.QrCodeScanner,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "Scan to pay",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun StatusSection(status: PaymentStatus) {
    AnimatedContent(
        targetState = status,
        transitionSpec = {
            (fadeIn() + slideInVertically { it }).togetherWith(fadeOut() + slideOutVertically { -it })
        },
        label = "Status Animation"
    ) { currentStatus ->
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            val (color, text) = when (currentStatus) {
                PaymentStatus.WAITING -> MaterialTheme.colorScheme.primary to "Waiting for scan..."
                PaymentStatus.MANDATE_PENDING -> Color(0xFFFFA000) to "Processing mandate..."
                PaymentStatus.AI_EVALUATION -> Color(0xFF1976D2) to "AI Risk Evaluation..."
                PaymentStatus.INSTALLMENT_PENDING -> Color(0xFF7B1FA2) to "Setting up installments..."
                PaymentStatus.COMPLETED -> Color(0xFF388E3C) to "Payment Successful!"
                PaymentStatus.TIMEOUT -> MaterialTheme.colorScheme.error to "Timed out"
                PaymentStatus.ERROR -> MaterialTheme.colorScheme.error to "Error occurred"
            }

            if (currentStatus != PaymentStatus.COMPLETED && currentStatus != PaymentStatus.ERROR && currentStatus != PaymentStatus.TIMEOUT) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    color = color,
                    strokeWidth = 3.dp
                )
                Spacer(modifier = Modifier.height(12.dp))
            }

            Text(
                text = text,
                style = MaterialTheme.typography.titleMedium,
                color = color,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}
