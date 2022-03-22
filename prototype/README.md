STOPPING ZOMBIE PROCESS WITH POWERSHELL: 

Stop-Process -id (Get-Process -Id (Get-NetTCPConnection -LocalPort 5555).OwningProcess).Id