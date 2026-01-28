<!-- README.md -->

# Project Template

This project is designed to be executed via shell scripts.

## How to Run

All commands below should be executed **from the project root directory**.

```bash
chmod +x spts/debug.sh
./spts/debug.sh
echo "bash /home/admin/desktop/username/spts/debug.sh" | at 09:12
```
```powershell
schtasks /Create /SC ONCE /TN "train_once" /TR "C:\Users\admin\Desktop\username\spts\debug.bat" /ST 11:21
```


<!-- end of README.md -->