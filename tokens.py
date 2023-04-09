PAD = 0x00 # Padding token - NUL Null
CTX = 0x01 # Context for generation - SOH Start of Heading
COT = 0x02 # Chain of Thought start - SOT Start of Text
SAY = 0x03 # Chain of Thought end, say the next text - ETX End of Text
SEP = 0x04 # Separator - EOT End of Transmission
REQ = 0x05 # Request an external tool (eg [REQ] calculator 1 + 1 [SEP] 2 [REQ]) - INQ Enquiry

UNUSED_06 = 0x06 # ACK Acknowledge
UNUSED_07 = 0x07 # BEL Bell
UNUSED_08 = 0x08 # BS  Backspace

# 0x09 HT Horizontal Tab is a common ASCII character
# 0x0A LF Line Feed is a common ASCII character

UNUSED_0B = 0x0B # VT  Vertical Tab
UNUSED_0C = 0x0C # FF  Form Feed

# 0x0D CR Carriage Return is a common ASCII character

UNUSED_0E = 0x0E # SO  Shift Out
UNUSED_0F = 0x0F # SI  Shift In
UNUSED_10 = 0x10 # DLE Data Link Escape
UNUSED_11 = 0x11 # DC1 Device Control 1
UNUSED_12 = 0x12 # DC2 Device Control 2
UNUSED_13 = 0x13 # DC3 Device Control 3
UNUSED_14 = 0x14 # DC4 Device Control 4
UNUSED_15 = 0x15 # NAK Negative Acknowledge
UNUSED_16 = 0x16 # SYN Synchronous Idle
UNUSED_17 = 0x17 # ETB End of Transmission Block

CAN = 0x18 # User issued cancel signal - CAN Cancel
END = 0x19 # Document ended - EM End of Medium
SUB = 0x1A # A UTF-8 coding error - SUB Substitute
ESC = 0x1B # Escape character, used for ANSI codes - ESC Escape

UNUSED_1C = 0x1C # FS  File Separator
UNUSED_1D = 0x1D # GS  Group Separator
UNUSED_1E = 0x1E # RS  Record Separator
UNUSED_1F = 0x1F # US  Unit Separator

# (0x20 - 0x7E normal ASCII)

DEL = 0x7F # DEL Delete
UNK = 0x7F # Unknown character, deletions for forgetful models

# (0x80 - 0xBF UTF-8 continuation bytes)

UNUSED_C0 = 0xC0 # Always codes for an overlong character
UNUSED_C1 = 0xC1 # Always codes for an overlong character

# (0xC2 - 0xF4 UTF-8 leading bytes)

UNUSED_F5 = 0xF5 # Codes for values larger than U+10FFFF
UNUSED_F6 = 0xF6 # Codes for values larger than U+10FFFF
UNUSED_F7 = 0xF7 # Codes for values larger than U+10FFFF
UNUSED_F8 = 0xF8 # Codes for values larger than U+10FFFF
UNUSED_F9 = 0xF9 # Codes for values larger than U+10FFFF
UNUSED_FA = 0xFA # Codes for values larger than U+10FFFF
UNUSED_FB = 0xFB # Codes for values larger than U+10FFFF
UNUSED_FC = 0xFC # Codes for values larger than U+10FFFF
UNUSED_FD = 0xFD # Codes for values larger than U+10FFFF
UNUSED_FE = 0xFE # No assigned meaning
UNUSED_FF = 0xFF # No assigned meaning