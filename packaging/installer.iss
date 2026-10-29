; Kinetra Motion Studio - Inno Setup Script
; Generates a professional Windows Setup Wizard with custom Mocap/Takes directory selection.

#define MyAppName "Kinetra Motion Studio"
#define MyAppVersion "0.4.0"
#define MyAppPublisher "Kinetra"
#define MyAppURL "https://github.com/maetarenn/kinetra"
#define MyAppExeName "Kinetra.exe"

[Setup]
AppId={{5E22B9D9-92F6-4608-8FD5-3C77E4A8F932}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\Kinetra
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=..\installer_output
OutputBaseFilename=Kinetra_Setup_v0.4.0
SetupIconFile=..\kinetra\assets\icon.ico
WizardSmallImageFile=wizard_small.bmp
WizardImageFile=wizard_large.bmp
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=commandline dialog
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
turkish.TakesPageTitle=Mocap ve Çekim (Takes) Klasörü
turkish.TakesPageSubTitle=Kayıt ve animasyonların kaydedileceği klasörü belirleyin
turkish.TakesPageDesc=Kinetra yüksek çözünürlüklü hareket yakalama ve kamera verilerini bu klasörde saklar.%nLütfen çekimlerinizin (takes) kaydedileceği varsayılan klasörü seçin:
turkish.TakesFolderPrompt=Mocap Kayıt Klasörü:
turkish.DesktopShortcut=Masaüstü simgesi oluştur
turkish.TakesShortcut=Mocap Çekimleri (Takes) Klasörünü Başlat Menüsüne Ekle

english.TakesPageTitle=MoCap & Takes Directory
english.TakesPageSubTitle=Select where recordings and takes will be stored
english.TakesPageDesc=Kinetra stores high-resolution motion capture and camera takes in this directory.%nPlease choose the default folder for your takes:
english.TakesFolderPrompt=Takes Storage Folder:
english.DesktopShortcut=Create a desktop shortcut
english.TakesShortcut=Add MoCap Takes Folder to Start Menu

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopShortcut}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "takesfolder"; Description: "{cm:TakesShortcut}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\Kinetra\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\kinetra\assets\icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\kinetra\assets\icon.ico"; Tasks: desktopicon
Name: "{group}\Mocap Çekimleri (Takes)"; Filename: "{code:GetTakesDir}"; Tasks: takesfolder
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  TakesDirPage: TInputDirWizardPage;

procedure InitializeWizard;
begin
  // Create custom directory selection page right after Select Destination Directory
  TakesDirPage := CreateInputDirPage(
    wpSelectDir,
    CustomMessage('TakesPageTitle'),
    CustomMessage('TakesPageSubTitle'),
    CustomMessage('TakesPageDesc'),
    False, 'Kinetra Captures');

  TakesDirPage.Add(CustomMessage('TakesFolderPrompt'));
  TakesDirPage.Values[0] := ExpandConstant('{userdocs}\Kinetra Captures');
end;

function GetTakesDir(Param: String): String;
begin
  Result := TakesDirPage.Values[0];
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  SelectedTakesDir: String;
begin
  if CurStep = ssPostInstall then
  begin
    SelectedTakesDir := TakesDirPage.Values[0];
    // Create the directory if it does not already exist
    ForceDirectories(SelectedTakesDir);
    // Register with QSettings for Kinetra
    RegWriteStringValue(HKEY_CURRENT_USER, 'Software\Kinetra\Kinetra', 'default_project_dir', SelectedTakesDir);
    RegWriteStringValue(HKEY_CURRENT_USER, 'Software\Kinetra\Kinetra', 'project', SelectedTakesDir + '\DefaultProject');
  end;
end;
