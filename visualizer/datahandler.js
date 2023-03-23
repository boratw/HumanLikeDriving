function RequestCurrentMap()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentMap(xmlHttp.responseText);            
        }
    };
    
    xmlHttp.open("GET", "/v/curstate/" + current_step, true);
    xmlHttp.send();

    var xmlHttp2 = new XMLHttpRequest();
    xmlHttp2.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentExp(xmlHttp2.responseText);            
        }
    };
    
    xmlHttp2.open("GET", "/v/expinfo/", true);
    xmlHttp2.send();
}
function RequestLatentResult()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleLatentResult(xmlHttp.responseText);            
        }
    };
    url = "/v/latentout/" + clicked + "/";
    for(var i = 0; i < 8; ++i)
        url += document.getElementById("value_l" + i).innerHTML + "/"
    xmlHttp.open("GET", url, true);
    xmlHttp.send();
}

function HandleCurrentMap(response)
{
    data = JSON.parse(response);
    vehicles = data["state"];
    routes = data["route"];
    latents = data["latent"];
    predicteds = data["predicted"];
    DrawCanvas();
}

function HandleCurrentExp(response)
{
    data = JSON.parse(response);
    max_step = data["max_step"];
    latent_len = data["latent_len"];

    document.getElementById("slider_step").max = max_step;
    for(var i = 0; i < 8; ++i)
    {
        if(i >= latent_len)
            document.getElementById("div_l" + i).hidden = true;
    }

}

function HandleLatentResult(response)
{
    data = JSON.parse(response);
    latentoutput = data["predicted"];
    DrawCanvas();
}