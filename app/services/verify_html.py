def render_verify_page(title: str, message: str, *, success: bool) -> str:
    accent = "#4ec9b0" if success else "#f48771"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} - LR-Agent</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      box-sizing: border-box;
      background: #1e1e1e;
      color: #cccccc;
      font-family: "Microsoft YaHei", "Source Han Sans SC", sans-serif;
    }}
    .card {{
      width: min(420px, 100%);
      padding: 28px;
      border-radius: 8px;
      border: 1px solid #3c3c3c;
      background: #252526;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
      text-align: center;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      color: {accent};
    }}
    p {{
      margin: 0;
      font-size: 15px;
      line-height: 1.6;
      color: #cccccc;
    }}
    .hint {{
      margin-top: 16px;
      font-size: 13px;
      color: #9d9d9d;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <p>{message}</p>
    <p class="hint">验证完成后，请回到 LR-Agent 客户端刷新账户资料。</p>
  </div>
</body>
</html>"""
