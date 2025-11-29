package com.morzio.pos.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.morzio.pos.ui.screens.AmountInputScreen
import com.morzio.pos.ui.screens.QRCodeScreen

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(navController = navController, startDestination = "amount") {
        composable("amount") {
            AmountInputScreen(onChargeClicked = {
                navController.navigate("qrcode/$it")
            })
        }
        composable(
            "qrcode/{amount}",
            arguments = listOf(navArgument("amount") { type = NavType.StringType })
        ) {
            QRCodeScreen(amount = it.arguments?.getString("amount"))
        }
    }
}