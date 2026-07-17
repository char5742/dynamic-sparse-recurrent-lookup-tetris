Set-StrictMode -Version Latest

function ConvertTo-NativeArgument([AllowEmptyString()][string]$Argument) {
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }
    $Builder = [System.Text.StringBuilder]::new()
    [void]$Builder.Append('"')
    $Backslashes = 0
    foreach ($Character in $Argument.ToCharArray()) {
        if ($Character -eq '\') {
            $Backslashes += 1
            continue
        }
        if ($Character -eq '"') {
            for ($Index = 0; $Index -lt (2 * $Backslashes + 1); $Index += 1) {
                [void]$Builder.Append('\')
            }
            [void]$Builder.Append('"')
        } else {
            for ($Index = 0; $Index -lt $Backslashes; $Index += 1) {
                [void]$Builder.Append('\')
            }
            [void]$Builder.Append($Character)
        }
        $Backslashes = 0
    }
    # Backslashes immediately before the closing quote must be doubled.
    for ($Index = 0; $Index -lt (2 * $Backslashes); $Index += 1) {
        [void]$Builder.Append('\')
    }
    [void]$Builder.Append('"')
    return $Builder.ToString()
}

function Join-NativeArguments([string[]]$ArgumentList) {
    return (($ArgumentList | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
}
