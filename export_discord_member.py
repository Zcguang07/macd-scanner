import os
import re
import json
import asyncio
import datetime
from pathlib import Path
import aiohttp
import discord

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
TARGET_CHANNEL_IDS = [
    int(x.strip()) for x in os.environ.get("TARGET_CHANNEL_IDS", "1307432516109598720").split(",") if x.strip()
]
TARGET_USER_KEYWORD = os.environ.get("TARGET_USER_KEYWORD", "")
OUTPUT_DIR = Path("exported_discord_notes")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)


def match_author(author: discord.abc.User, keyword: str) -> bool:
    if not keyword:
        return True
    kw = keyword.strip().lower()
    if str(author.id) == kw:
        return True
    for attr in ("name", "display_name", "global_name"):
        val = getattr(author, attr, None)
        if val and kw in val.lower():
            return True
    return False


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|\s]+', "_", name)


async def download_file(session: aiohttp.ClientSession, url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    for attempt in range(3):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    dest_path.write_bytes(data)
                    return True
        except Exception as e:
            if attempt == 2:
                print(f"  [!] 下载图片失败 ({url}): {e}")
            await asyncio.sleep(1.5)
    return False


async def export_channel(channel: discord.abc.Messageable, session: aiohttp.ClientSession):
    ch_name = safe_filename(getattr(channel, "name", str(channel.id)))
    ch_dir = OUTPUT_DIR / ch_name
    img_dir = ch_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始全量抓取频道: #{ch_name} (ID: {channel.id})")

    md_blocks = []
    html_blocks = []
    raw_records = []
    total_scanned = 0
    matched_count = 0
    downloaded_imgs = 0

    async for msg in channel.history(limit=None, oldest_first=True):
        total_scanned += 1
        if total_scanned % 500 == 0:
            print(f"[{ch_name}] 已扫描 {total_scanned} 条 | 命中 {matched_count} 条 | 已存原图 {downloaded_imgs} 张...")

        if not match_author(msg.author, TARGET_USER_KEYWORD):
            continue

        content_text = (msg.content or "").strip()
        embed_texts = []
        for emb in msg.embeds:
            parts = []
            if emb.title:
                parts.append(f"**{emb.title}**")
            if emb.description:
                parts.append(emb.description)
            for fld in emb.fields:
                parts.append(f"- **{fld.name}**: {fld.value}")
            if parts:
                embed_texts.append("\n".join(parts))

        if not content_text and not msg.attachments and not embed_texts:
            continue

        matched_count += 1
        bj_dt = msg.created_at + datetime.timedelta(hours=8)
        bj_str = bj_dt.strftime("%Y-%m-%d %H:%M:%S")
        date_prefix = bj_dt.strftime("%Y%m%d_%H%M%S")
        author_name = getattr(msg.author, "display_name", msg.author.name)

        reply_md = ""
        reply_html = ""
        if msg.reference and msg.reference.resolved and isinstance(msg.reference.resolved, discord.Message):
            ref = msg.reference.resolved
            ref_author = getattr(ref.author, "display_name", ref.author.name)
            ref_content = (ref.content or "[图片/卡片]").replace("\n", " ")
            reply_md = f"> 💬 **回复 @{ref_author}**：{ref_content}\n\n"
            reply_html = f'<div class="reply">💬 <b>回复 @{ref_author}</b>：{ref_content}</div>'

        local_images_md = []
        local_images_html = []
        saved_files = []

        for idx, att in enumerate(msg.attachments, start=1):
            ext = Path(att.filename).suffix or ".png"
            fname = f"{date_prefix}_{msg.id}_{idx}{ext}"
            fpath = img_dir / fname
            rel_path = f"images/{fname}"

            ok = await download_file(session, att.url, fpath)
            if ok:
                downloaded_imgs += 1
                saved_files.append(rel_path)
                is_img = (att.content_type and att.content_type.startswith("image/")) or ext.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp")
                if is_img:
                    local_images_md.append(f"![{att.filename}]({rel_path})")
                    local_images_html.append(f'<div class="img-box"><a href="{rel_path}" target="_blank"><img src="{rel_path}" loading="lazy"></a></div>')
                else:
                    local_images_md.append(f"- 📎 附件：[{att.filename}]({rel_path})")
                    local_images_html.append(f'<div>📎 <a href="{rel_path}" target="_blank">{att.filename}</a></div>')
            else:
                local_images_md.append(f"![图片下载失败_原链接]({att.url})")

        md_piece = [f"### [{bj_str}] {author_name}"]
        if reply_md:
            md_piece.append(reply_md.strip())
        if content_text:
            md_piece.append(content_text)
        if embed_texts:
            md_piece.extend(embed_texts)
        if local_images_md:
            md_piece.extend(local_images_md)
        md_blocks.append("\n\n".join(md_piece))

        html_content = content_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html_piece = f"""
        <div class="msg-card">
          <div class="msg-header"><span class="author">{author_name}</span> <span class="time">{bj_str}</span></div>
          {reply_html}
          <div class="msg-body">{html_content}</div>
          {''.join(local_images_html)}
        </div>
        """
        html_blocks.append(html_piece)

        raw_records.append({
            "id": str(msg.id),
            "time": bj_str,
            "author": author_name,
            "reply_to": reply_md.strip(),
            "content": content_text,
            "images": saved_files,
        })

    md_file = ch_dir / f"{ch_name}_完整图文笔记.md"
    md_header = [
        f"归档频道：#{ch_name} ｜ 筛选成员：{TARGET_USER_KEYWORD or '全部成员'}",
        f"统计：共扫描 {total_scanned} 条消息 ｜ 命中导出 {matched_count} 条发言 ｜ 离线保存高清原图 {downloaded_imgs} 张",
        "",
    ]
    md_file.write_text("\n".join(md_header) + "\n\n" + "\n\n".join(md_blocks) + "\n", encoding="utf-8")

    html_file = ch_dir / f"{ch_name}_离线看图学习版.html"
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{ch_name} - 离线图文学习笔记</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; background: #1e1f22; color: #dbdee1; max-width: 920px; margin: 0 auto; padding: 24px; line-height: 1.65; }}
  h1 {{ color: #fff; border-bottom: 2px solid #4ECABA; padding-bottom: 10px; }}
  .summary {{ color: #4ECABA; margin-bottom: 24px; font-weight: 600; }}
  .msg-card {{ background: #2b2d31; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; border: 1px solid #3f4147; }}
  .msg-header {{ margin-bottom: 8px; }}
  .author {{ color: #4ECABA; font-weight: 700; font-size: 16px; margin-right: 10px; }}
  .time {{ color: #949ba4; font-size: 13px; }}
  .reply {{ background: #232428; border-left: 3px solid #4ECABA; padding: 6px 12px; color: #b5bac1; font-size: 13px; margin-bottom: 10px; border-radius: 0 4px 4px 0; }}
  .msg-body {{ font-size: 15px; color: #f2f3f5; word-break: break-word; }}
  .img-box {{ margin-top: 12px; }}
  .img-box img {{ max-width: 100%; max-height: 520px; border-radius: 8px; border: 1px solid #4e5058; cursor: zoom-in; }}
</style>
</head>
<body>
<h1>#{ch_name} · 交易实战图文全集</h1>
<div class="summary">共命中 {matched_count} 条干货发言 · 已离线保存 {downloaded_imgs} 张高清原图（点击图片可放大查看原图）</div>
{''.join(html_blocks)}
</body>
</html>"""
    html_file.write_text(html_doc, encoding="utf-8")

    json_file = ch_dir / f"{ch_name}_raw_data.json"
    json_file.write_text(json.dumps(raw_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"频道 #{ch_name} 导出完毕！共命中 {matched_count} 条发言，保存原图 {downloaded_imgs} 张。")


@client.event
async def on_ready():
    print(f"登录成功: {client.user}")
    async with aiohttp.ClientSession() as session:
        for cid in TARGET_CHANNEL_IDS:
            ch = client.get_channel(cid)
            if ch is None:
                try:
                    ch = await client.fetch_channel(cid)
                except Exception as e:
                    print(f"[!] 无法访问频道 {cid}: {e}")
                    continue
            await export_channel(ch, session)
    await client.close()


if __name__ == "__main__":
    client.run(BOT_TOKEN)
