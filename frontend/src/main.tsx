import React from "react";
import ReactDOM from "react-dom/client";
import DemoApp from "./demo/DemoApp";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <DemoApp />
  </React.StrictMode>,
);
