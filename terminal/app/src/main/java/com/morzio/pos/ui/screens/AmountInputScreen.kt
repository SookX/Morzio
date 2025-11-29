package com.morzio.pos.ui.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.morzio.pos.viewmodels.AmountViewModel

@Composable
fun AmountInputScreen(onChargeClicked: (String) -> Unit) {
    val viewModel: AmountViewModel = viewModel()
    val uiState by viewModel.uiState.collectAsState()
    val haptic = LocalHapticFeedback.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp, 48.dp, 16.dp, 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(text = "Morzio Terminal", fontSize = 20.sp)
            Text(text = "Merchant ID: 123-456-789", fontSize = 12.sp, color = Color.Gray)
        }

        AmountDisplay(amount = uiState.amount)

        NumericKeypad(
            onKeyPress = {
                viewModel.onKeyPress(it)
            },
            decimalEnabled = false
        )

        ChargeButton(
            amount = uiState.amount,
            enabled = uiState.isValid,
            onClick = { onChargeClicked(uiState.amount) }
        )
    }
}

@Composable
fun AmountDisplay(amount: String) {
    val amountValue = amount.replace(",", ".").toDoubleOrNull() ?: 0.0
    val color by animateColorAsState(
        targetValue = if (amountValue > 0) MaterialTheme.colorScheme.primary else Color.Gray,
        label = ""
    )
    val scale by animateFloatAsState(targetValue = if (amount.isEmpty()) 0.9f else 1f, label = "")

    Text(
        text = "€${amount}",
        fontSize = 64.sp,
        color = color,
        textAlign = TextAlign.Center,
        modifier = Modifier.scale(scale)
    )
}

@Composable
fun NumericKeypad(onKeyPress: (String) -> Unit, decimalEnabled: Boolean) {
    val keys = listOf(
        "1", "2", "3",
        "4", "5", "6",
        "7", "8", "9",
        ".", "0", "←"
    )

    Column(modifier = Modifier.fillMaxWidth()) {
        keys.chunked(3).forEach { row ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                row.forEach { key ->
                    FilledTonalButton(
                        onClick = { onKeyPress(key) },
                        modifier = Modifier.size(80.dp),
                        enabled = if (key == ".") decimalEnabled else true
                    ) {
                        if (key == ".") {
                            Text(text = key, fontSize = 24.sp, color = Color.Transparent)
                        } else {
                            Text(text = key, fontSize = 24.sp)
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
fun ChargeButton(amount: String, enabled: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .height(60.dp),
        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
    ) {
        Text(
            text = if (enabled) "Charge Customer - €$amount" else "Charge Customer",
            fontSize = 20.sp
        )
    }
}