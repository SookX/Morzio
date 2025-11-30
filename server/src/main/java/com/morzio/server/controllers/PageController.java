package com.morzio.server.controllers;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class PageController {

    @GetMapping("/success")
    public String successPage() {
        return "success";
    }

    @GetMapping("/error")
    public String errorPage() {
        return "error";
    }
}
