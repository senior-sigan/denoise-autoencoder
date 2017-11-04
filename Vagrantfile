# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "google/gce"

  config.vm.provider :google do |google, override|
    google.google_project_id = "mltest-180907"
    google.google_client_email = "849715628762-compute@developer.gserviceaccount.com
eccac1efb12019dcbc29eab30b9174bf238864f0"
    google.google_json_key_location = "/Volumes/Macintosh HD/Yandex.Disk.localized/mltest-eccac1efb120.json"

    override.ssh.username = "ilya.blan4"
    override.ssh.private_key_path = "~/.ssh/id_rsa"
    #override.ssh.private_key_path = "~/.ssh/google_compute_engine"
    
    google.zone = "europe-west1-b"
    google.zone_config "europe-west1-b" do |zone|
        zone.name = "ml-instance"
        zone.image = "ubuntu-1604-xenial-v20171026a"
        zone.image_family = "ubuntu-1604-lts"
        zone.machine_type = "n1-standard-1"
        zone.zone = "europe-west1-b"
        zone.tags = ['jupyter', 'tensorflow', 'http-server']
        zone.auto_restart = false
        zone.on_host_maintenance = 'TERMINATE'
        zone.can_ip_forward = true
    end
  end

  config.vm.provision :shell, path: 'bootstrap.sh', keep_color: true
  
  config.vm.synced_folder ".", "/vagrant", disabled: true
  config.vm.synced_folder "./src", "/src", owner: "root", group: "root", :mount_options => ["dmode=777", "fmode=666"]
  config.vm.synced_folder "./data/spectrums", "/src/data/spectrums", owner: "root", group: "root", :mount_options => ["dmode=777", "fmode=666"]
end
