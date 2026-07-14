import qrcode
from pathlib import Path
import sys
from hashlib import sha256
import re
from urllib.parse import urlsplit

def generate_secure_qr():
    """
    Phase 5: Secure QR Code Generation (Strict Production URL)
    แปลง URL เป็น QR Code และตั้งชื่อไฟล์ตาม URL
    """
    server_url = "https://praise.streamlit.app/"
    
    # Strict URL Validation (Zero-Silent Bug Policy)
    parsed = urlsplit(server_url)
    
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("Zero-Silent Bug: The provided base URL is invalid. It must be an absolute HTTP(S) URL with a hostname.")
        
    if parsed.username or parsed.password:
        raise RuntimeError("Zero-Silent Bug: The provided base URL must not contain credentials (username/password).")
        
    try:
        _ = parsed.port
    except ValueError:
        raise RuntimeError("Zero-Silent Bug: The provided base URL contains an invalid port.")
        
    # ทำความสะอาด URL และเพิ่ม Hash เพื่อป้องกันชื่อไฟล์ซ้ำและรองรับ URL ยาว (Collision-resistant)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", server_url).strip("_")[:80] or "target"
    url_hash = sha256(server_url.encode("utf-8")).hexdigest()[:12]
    output_filename = f"PRAISE_QR_{safe_name}_{url_hash}.png"
    
    # ตั้งค่าความละเอียดและ Error Correction ระดับสูง (High) 
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    
    qr.add_data(server_url)
    qr.make(fit=True)
    
    # สร้างภาพพื้นหลังขาว รูปคิวอาร์โค้ดสีดำ
    img = qr.make_image(fill_color="black", back_color="white")
    
    # บันทึกไฟล์ลงในโฟลเดอร์ root ของโปรเจกต์
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_filename
    img.save(output_path)
    
    print(f"Phase 5 Complete: Secure QR Code generated successfully at: {output_path}")
    print(f"Target host: {parsed.hostname}")

if __name__ == "__main__":
    try:
        generate_secure_qr()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
